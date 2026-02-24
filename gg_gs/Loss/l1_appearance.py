import torch

from base_gs_trainer.Loss.l1 import l1_loss

from gg_gs.Model.gs import GaussianModel


def L1_loss_appearance(image, gt_image, gaussians, view_idx):
    app_model = gaussians.app_model
    if app_model is GaussianModel.App_model.NO:
        return l1_loss(image, gt_image)
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)

    if app_model is GaussianModel.App_model.GS:
        exposure = appearance_embedding
        transformed = torch.addmm(
            exposure[:3, 3, None],
            exposure[:3, :3],
            image.reshape(3, -1),
        ).reshape_as(image)
        return l1_loss(transformed, gt_image)

    if app_model is GaussianModel.App_model.GOF:
        origH, origW = image.shape[1:]
        H, W = origH // 32 * 32, origW // 32 * 32
        top, left = (origH - H) // 2, (origW - W) // 2

        crop = image[:, top : top + H, left : left + W]
        crop_gt = gt_image[:, top : top + H, left : left + W]

        down = torch.nn.functional.interpolate(crop[None], size=(H // 32, W // 32), mode="bilinear", align_corners=True)[0]
        embedding_map = appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)
        net_in = torch.cat([down, embedding_map], dim=0)[None]

        mapping = gaussians.appearance_network(net_in)
        transformed = mapping * crop
        return l1_loss(transformed, crop_gt)

    if app_model is GaussianModel.App_model.PGSR:
        transformed = torch.addcmul(appearance_embedding[1], torch.exp(appearance_embedding[0]), image)
        return l1_loss(transformed, gt_image)
