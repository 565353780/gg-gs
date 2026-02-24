from base_gs_trainer.Config.config import (
    ParamGroup,
    BaseModelParams,
    BasePipelineParams,
    BaseOptimizationParams,
)


class ModelParams(BaseModelParams, ParamGroup):
    def __init__(self, parser, sentinel=False):
        BaseModelParams.__init__(self)

        self.sg_degree = 0
        self._dataset = ""
        self.use_decoupled_appearance = 3 # 0: NO, 1: GS, 2: GOF, 3: PGSR
        self.disable_filter3D = False
        self.kernel_size = 0.0 # Size of 2D filter in mip-splatting

        self.multi_view_num = 8
        self.multi_view_max_angle = 30
        self.multi_view_min_dis = 0.01
        self.multi_view_max_dis = 1.5

        ParamGroup.__init__(self, parser, "Loading Parameters", sentinel)
        return

class PipelineParams(BasePipelineParams, ParamGroup):
    def __init__(self, parser):
        BasePipelineParams.__init__(self)

        ParamGroup.__init__(self, parser, "Pipeline Parameters")
        return

class OptimizationParams(BaseOptimizationParams, ParamGroup):
    def __init__(self, parser):
        BaseOptimizationParams.__init__(self)
        self.feature_dc_lr = 0.0013
        self.feature_rest_lr = 0.00011

        self.opacity_lr = 0.05

        self.sg_axis_lr = 0.002
        self.sg_sharpness_lr = 0.095
        self.sg_color = 0.00064
        self.appearance_embeddings_lr = 0.001
        self.appearance_network_lr = 0.001
        self.pgsr_appearance_lr = 0.001
        self.gs_appearance_lr_init = 0.01
        self.gs_appearance_lr_final = 0.001
        self.gs_appearance_lr_delay_steps = 0
        self.gs_appearance_lr_delay_mult = 0.0

        self.percent_dense = 0.01

        self.lambda_depth_normal = 0.05

        self.densification_interval = 100
        #self.opacity_reset_interval = 500
        self.opacity_reset_interval = 3000
        self.scaling_reset_interval = 500
        self.densify_until_iter = 15_000
        self.regularization_from_iter = 7000
        self.densify_grad_threshold = 0.0002

        self.lambda_multi_view_geo = 0.02
        self.lambda_multi_view_ncc = 0.6
        self.multi_view_patch_size = 3
        self.multi_view_pixel_noise_th = 1.0
        self.use_geo_occ_aware = True

        ParamGroup.__init__(self, parser, "Optimization Parameters")
        return
