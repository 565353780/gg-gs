import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

import warp_patch_ncc

from base_gs_trainer.Data.gs_camera import GSCamera

from utils.graphics_utils import patch_offsets

from gg_gs.Model.gs import GaussianModel
from gg_gs.Method.render_kernel import sample_depth


class PatchMatch:
    def __init__(self, patch_size, pixel_noise_th, kernel_size, pipe, debug=True, model_path=None):
        self.patch_size = patch_size
        self.total_patch_size = (patch_size * 2 + 1) ** 2
        self.pixel_noise_th = pixel_noise_th
        self.offsets = patch_offsets(patch_size, device="cuda") * 0.5
        self.offsets.requires_grad_(False)
        self.kernel_size = kernel_size
        self.pipe = pipe
        self.debug = debug
        self.model_path = model_path
        if debug:
            os.makedirs(os.path.join(model_path, "debug"), exist_ok=True)

    def __call__(self, gaussians: GaussianModel, render_pkg: dict, viewpoint_cam: GSCamera, nearest_cam: GSCamera, iteration=0, depth_normal=None):
        if nearest_cam is None:
            return torch.tensor([0], dtype=torch.float32, device="cuda"), torch.tensor([0], dtype=torch.float32, device="cuda")
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        ## compute geometry consistency mask
        with torch.no_grad():
            ix = (torch.arange(W, device="cuda", dtype=torch.float32) - viewpoint_cam.Cx) / viewpoint_cam.Fx
            iy = (torch.arange(H, device="cuda", dtype=torch.float32) - viewpoint_cam.Cy) / viewpoint_cam.Fy
            view_to_nearest_T = (
                -viewpoint_cam.world_view_transform[:3, :3].T @ nearest_cam.R @ nearest_cam.T + viewpoint_cam.world_view_transform[3, :3]
            )
            nearest_to_view_R = nearest_cam.R.transpose(1, 0) @ viewpoint_cam.world_view_transform[:3, :3]

        # pts = (rays_d * render_pkg["median_depth"].squeeze().unsqueeze(-1)).reshape(-1, 3)
        depth_reshape = render_pkg["median_depth"].squeeze().unsqueeze(-1)
        pts = torch.cat([depth_reshape * ix[None, :, None], depth_reshape * iy[:, None, None], depth_reshape], dim=-1)

        R = viewpoint_cam.R
        T = viewpoint_cam.T
        pts = (pts - T) @ R.T
        sampled_pkg = sample_depth(
            pts,
            nearest_cam,
            gaussians,
            self.pipe,
            self.kernel_size,
        )

        pts_in_nearest_cam = sampled_pkg["sampled_depth"]
        R = nearest_cam.R
        T = nearest_cam.T

        pts_in_view_cam = view_to_nearest_T + pts_in_nearest_cam @ nearest_to_view_R
        pts_projections = pts_in_view_cam[..., :2] / torch.clamp_min(pts_in_view_cam[..., 2:], 1e-7)
        pts_projections = torch.addcmul(
            pts_projections.new_tensor([viewpoint_cam.Cx, viewpoint_cam.Cy]),
            pts_projections.new_tensor([viewpoint_cam.Fx, viewpoint_cam.Fy]),
            pts_projections,
        )

        ix, iy = torch.meshgrid(
            torch.arange(W, device="cuda", dtype=torch.int32),
            torch.arange(H, device="cuda", dtype=torch.int32),
            indexing="xy",
        )
        pixels = torch.stack([ix, iy], dim=-1)
        pixel_f = pixels.type(torch.float32).requires_grad_(False)
        pixel_noise = torch.pairwise_distance(pts_projections, pixel_f)

        with torch.no_grad():
            d_mask = (
                sampled_pkg["inside"]
                & (pts_in_nearest_cam[..., -1] > 0.2)
                & (pts_in_view_cam[..., -1] > 0.2)
                & (pixel_noise < self.pixel_noise_th)
                & (render_pkg["median_depth"].squeeze() > 0)
            )
            weights = torch.exp(-pixel_noise)
            weights[~d_mask] = 0
        ##############################################

        if iteration % 200 == 0 and self.debug:
            with torch.no_grad():
                gt_img_show = (viewpoint_cam.original_image.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                img_show = ((render_pkg["render"]).permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                normal_show = (((render_pkg["normal"] + 1.0) * 0.5).permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
                if depth_normal is None:
                    depth_normal_show = (
                        (nearest_cam.original_image.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                    )
                else:
                    depth_normal_show = (((depth_normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
                d_mask_show = (weights.float() * 255).detach().cpu().numpy().astype(np.uint8)
                d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                depth = render_pkg["median_depth"].squeeze().detach().cpu().numpy()
                depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                row0 = np.concatenate([gt_img_show, img_show, depth_normal_show], axis=1)
                row1 = np.concatenate([d_mask_show_color, depth_color, normal_show], axis=1)
                image_to_show = np.concatenate([row0, row1], axis=0)
                cv2.imwrite(os.path.join(self.model_path, "debug", "%05d" % iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)
        ################## Compute NCC for warped patches ##################
        if not d_mask.any():
            return torch.tensor([0], dtype=torch.float32, device="cuda"), torch.tensor([0], dtype=torch.float32, device="cuda")

        geo_loss = ((weights * pixel_noise)[d_mask]).mean()
        with torch.no_grad():
            d_mask = torch.flatten(d_mask)
            valid_indices = torch.argwhere(d_mask).squeeze(1)
            weights = torch.flatten(weights)[valid_indices]
            pixels = torch.index_select(pixels.view(-1, 2), dim=0, index=valid_indices)
            ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1, -2) @ viewpoint_cam.world_view_transform[:3, :3]
            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3, :3] + nearest_cam.world_view_transform[3, :3]

        depth_select = torch.index_select(render_pkg["median_depth"].view(-1), dim=0, index=valid_indices)
        normal_select = torch.index_select(render_pkg["normal"].view(3, -1), dim=1, index=valid_indices).transpose(1, 0)
        normal_select = F.normalize(normal_select, dim=-1)

        cc, valid_mask = warp_patch_ncc.warp_patch_ncc(
            depth_select,
            normal_select,
            pixels,
            ref_to_neareast_r.T,
            ref_to_neareast_t,
            viewpoint_cam.gray_image.to("cuda").squeeze(),
            nearest_cam.gray_image.to("cuda").squeeze(),
            viewpoint_cam.Fx,
            viewpoint_cam.Fy,
            viewpoint_cam.Cx,
            viewpoint_cam.Cy,
            nearest_cam.Fx,
            nearest_cam.Fy,
            nearest_cam.Cx,
            nearest_cam.Cy,
            False,
        )
        ncc = torch.clamp(1 - cc, 0.0, 2.0)
        ncc_mask = (ncc < 0.9) & valid_mask

        ncc = ncc.squeeze() * weights
        ncc = ncc[ncc_mask.squeeze()]

        if ncc_mask.any():
            ncc_loss = ncc.mean()
        else:
            ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        return ncc_loss, geo_loss
