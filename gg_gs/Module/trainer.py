import os
import sys
import cv2
import json
import torch
import numpy as np

from torch import nn
from typing import Tuple
from random import sample
from tqdm import tqdm, trange
from argparse import ArgumentParser

from fused_ssim import fused_ssim

from utils.graphics_utils import depth_to_normal

from base_gs_trainer.Module.base_gs_trainer import BaseGSTrainer

from gg_gs.Config.config import ModelParams, PipelineParams, OptimizationParams
from gg_gs.Loss.patch_match import PatchMatch
from gg_gs.Loss.l1_appearance import L1_loss_appearance
from gg_gs.Method.render_kernel import render
from gg_gs.Model.gs import GaussianModel


class Trainer(BaseGSTrainer):
    def __init__(
        self,
        colmap_data_folder_path: str='',
        device: str='cuda:0',
        save_result_folder_path: str='./output/',
        save_log_folder_path: str='./logs/',
        test_freq: int=10000,
        save_freq: int=10000,
    ) -> None:
        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args(sys.argv[1:])

        args.source_path = colmap_data_folder_path
        args.model_path = save_result_folder_path

        print("Optimizing " + args.model_path)

        self.dataset = lp.extract(args)
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

        self.gaussians = GaussianModel(self.dataset.sh_degree, self.dataset.sg_degree)

        self.args = args
        os.makedirs(os.path.join(self.dataset.model_path, "debug"), exist_ok=True)

        BaseGSTrainer.__init__(
            self,
            colmap_data_folder_path=colmap_data_folder_path,
            device=device,
            save_result_folder_path=save_result_folder_path,
            save_log_folder_path=save_log_folder_path,
            test_freq=test_freq,
            save_freq=save_freq,
        )

        with torch.no_grad():
            for camera_center in self.camera_centers_list:
                dists_cam_gauss = torch.norm(self.gaussians.get_xyz - camera_center[None, :], dim=1)
                max_scale = 0.05 * dists_cam_gauss.flatten()
                log_max_scale = torch.log(max_scale).repeat(3, 1).permute(1, 0)
                self.gaussians._scaling[:] = torch.clamp_max(self.gaussians._scaling, log_max_scale)


        self.update3DFilter()

        if self.opt.lambda_multi_view_ncc > 0 or self.opt.lambda_multi_view_geo > 0:
            self.patchmatch = PatchMatch(
                self.opt.multi_view_patch_size,
                self.opt.multi_view_pixel_noise_th,
                kernel_size=self.dataset.kernel_size,
                pipe=self.pipe,
                debug=True,
                model_path=self.dataset.model_path,
            )

        self.pipe.debug = True
        return

    def initGaussiansFromScene(self) -> bool:
        print("computing nearest_id")
        self.camera_centers_list: list[torch.Tensor] = []
        center_rays_list: list[torch.Tensor] = []
        with torch.no_grad():
            for id, cur_cam in enumerate(self.scene.train_cameras):
                self.camera_centers_list.append(cur_cam.camera_center)
                R = cur_cam.R
                center_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda()
                center_ray = center_ray @ R.transpose(-1, -2)
                center_rays_list.append(center_ray)
            camera_centers = torch.stack(self.camera_centers_list, dim=0)
            center_rays = torch.stack(center_rays_list, dim=0)
            center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
            diss = torch.norm(camera_centers[:, None] - camera_centers[None], dim=-1).detach().cpu().numpy()
            tmp = torch.sum(center_rays[:, None] * center_rays[None], dim=-1)
            angles_torch = torch.arccos(tmp) * 180 / 3.14159
            angles_np = angles_torch.detach().cpu().numpy()
            with open(os.path.join(self.dataset.model_path, "multi_view.json"), "w") as file:
                for id, cur_cam in enumerate(self.scene.train_cameras):
                    sorted_indices = np.lexsort((angles_np[id], diss[id]))
                    # sorted_indices = np.lexsort((diss[id], angles[id]))
                    mask = (
                        (angles_np[id][sorted_indices] < self.args.multi_view_max_angle)
                        & (diss[id][sorted_indices] > self.args.multi_view_min_dis)
                        & (diss[id][sorted_indices] < self.args.multi_view_max_dis)
                    )
                    sorted_indices = sorted_indices[mask]
                    multi_view_num = min(self.args.multi_view_num, len(sorted_indices))
                    json_d = {"ref_name": cur_cam.image_name, "nearest_name": []}
                    for index in sorted_indices[:multi_view_num]:
                        cur_cam.nearest_id.append(index)
                        # cur_cam.nearest_names.append(self.train_cameras[resolution_scale][index].image_name)
                        json_d["nearest_name"].append(self.scene.train_cameras[index].image_name)
                    json_str = json.dumps(json_d, separators=(",", ":"))
                    file.write(json_str)
                    file.write("\n")

        self.gaussians.create_app_model(len(self.scene), self.args.use_decoupled_appearance)
        return True

    def renderImage(
        self,
        viewpoint_cam,
        reg_kick_on: bool = False,
    ) -> dict:
        return render(
            viewpoint_cam,
            self.gaussians,
            self.pipe,
            self.background,
            self.dataset.kernel_size,
            require_depth=reg_kick_on,
        )

    def trainStep(
        self,
        iteration: int,
        viewpoint_cam,
        lambda_dssim: float = 0.2,
        lambda_opacity: float = 1e-6,
        lambda_scaling: float = 1.0,
    ) -> Tuple[dict, dict]:
        self.gaussians.update_learning_rate(iteration)


        if iteration % 1000 == 0:
            self.gaussians.unlockSGdegree(100)
            self.gaussians.oneupSHdegree()

        reg_kick_on = iteration >= self.opt.regularization_from_iter

        render_pkg = self.renderImage(viewpoint_cam, reg_kick_on)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()

        if reg_kick_on and self.opt.lambda_depth_normal > 0:
            depth_map: torch.Tensor = render_pkg["median_depth"]
            rendered_normal: torch.Tensor = render_pkg["normal"]
            depth_normal, valid_points = depth_to_normal(viewpoint_cam, depth_map)
            normal_error_map = 1 - torch.linalg.vecdot(rendered_normal, depth_normal, dim=0)
            depth_normal_loss = torch.where(valid_points.squeeze(), normal_error_map, torch.zeros_like(normal_error_map)).mean()
        else:
            depth_normal = None
            depth_normal_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        if reg_kick_on and (self.opt.lambda_multi_view_ncc > 0 or self.opt.lambda_multi_view_geo):
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else self.scene.train_cameras[sample(viewpoint_cam.nearest_id, 1)[0]]
            ncc_loss, geo_loss = self.patchmatch(self.gaussians, render_pkg, viewpoint_cam, nearest_cam, iteration, depth_normal)
        else:
            ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            geo_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        #reg_loss = l1_loss(image, gt_image)
        reg_loss = L1_loss_appearance(image, gt_image, self.gaussians, viewpoint_cam.uid)
        ssim_loss = 1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), padding="valid")
        rgb_loss = (1.0 - lambda_dssim) * reg_loss + lambda_dssim * ssim_loss

        opacity_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_opacity > 0:
            opacity_loss = nn.MSELoss()(self.gaussians.get_opacity, torch.zeros_like(self.gaussians._opacity))

        scaling_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_scaling > 0:
            scaling_loss = nn.MSELoss()(self.gaussians.get_scaling, torch.zeros_like(self.gaussians._scaling))

        # loss
        total_loss = rgb_loss + \
            self.opt.lambda_depth_normal * depth_normal_loss + \
            self.opt.lambda_multi_view_ncc * ncc_loss + \
            self.opt.lambda_multi_view_geo * geo_loss + \
            lambda_opacity * opacity_loss + \
            lambda_scaling * scaling_loss

        total_loss.backward()

        loss_dict = {
            'reg': reg_loss.item(),
            'ssim': ssim_loss.item(),
            'rgb': rgb_loss.item(),
            'depth_normal': depth_normal_loss.item(),
            'ncc': ncc_loss.item(),
            'geo': geo_loss.item(),
            'opacity': opacity_loss.item(),
            'scaling': scaling_loss.item(),
            'total': total_loss.item(),
        }

        return render_pkg, loss_dict

    @torch.no_grad
    def logImageStep(
        self,
        iteration: int,
        render_image_num: int=5,
        is_fast: bool=True,
    ) -> bool:
        BaseGSTrainer.logImageStep(self, iteration, render_image_num, is_fast)

        torch.cuda.empty_cache()

        for idx in trange(render_image_num):
            viewpoint = self.scene[idx]

            render_dict = self.renderImage(viewpoint, reg_kick_on=True)

            if self.logger.isValid():
                depth = render_dict["median_depth"].squeeze().detach().cpu().numpy()
                depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                self.logger.summary_writer.add_images(
                    "view_{}/depth".format(viewpoint.image_name),
                    depth_color.transpose(2, 0, 1)[None],
                    global_step=iteration,
                )
        return True

    @torch.no_grad()
    def recordGrads(self, render_pkg: dict) -> bool:
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        return True

    @torch.no_grad()
    def densifyStep(self, iteration: int) -> bool:
        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
        self.gaussians.densify_and_prune(
            self.opt.densify_grad_threshold,
            0.05,
            self.scene.cameras_extent,
            size_threshold,
        )
        return True

    @torch.no_grad()
    def update3DFilter(self) -> bool:
        if self.dataset.disable_filter3D:
            self.gaussians.reset_3D_filter()
        else:
            self.gaussians.compute_3D_filter(cameras=self.scene.train_cameras)
        return True

    @torch.no_grad()
    def resetOpacity(self) -> bool:
        self.gaussians.reset_opacity()
        return True

    @torch.no_grad()
    def resetScaling(self) -> bool:
        self.gaussians.reset_scaling()
        return True

    @torch.no_grad()
    def updateGSParams(self) -> bool:
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none = True)
        return True

    @torch.no_grad()
    def saveScene(self, iteration: int) -> bool:
        point_cloud_path = os.path.join(self.dataset.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        return True

    def train(self, iteration_num: int = 30000):
        progress_bar = tqdm(desc="Training progress", total=iteration_num)
        iteration = 0
        for _ in range(iteration_num):
            iteration += 1

            viewpoint_cam = self.scene[iteration]

            render_pkg, loss_dict = self.trainStep(iteration, viewpoint_cam)

            with torch.no_grad():
                if iteration % 10 == 0:
                    bar_loss_dict = {
                        "rgb": f"{loss_dict['rgb']:.{5}f}",
                        "Points": f"{len(self.gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(bar_loss_dict)
                    progress_bar.update(10)

                    self.logStep(iteration, loss_dict)

                if iteration % self.test_freq == 0:
                    self.logImageStep(
                        iteration,
                        render_image_num=1,
                        is_fast=True,
                    )

                if iteration % self.save_freq == 0:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    self.saveScene(iteration)

                # Densification
                if iteration < self.opt.densify_until_iter:
                    self.recordGrads(render_pkg)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        self.densifyStep(iteration)
                        self.update3DFilter()

                    if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.resetOpacity()

                    if iteration % self.opt.scaling_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.resetScaling()

                if iteration % 100 == 0 and iteration > self.opt.densify_until_iter and not self.dataset.disable_filter3D:
                    self.gaussians.compute_3D_filter(cameras=self.scene.train_cameras)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

            self.iteration = iteration
        return True
