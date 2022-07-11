# A script to visualize the ERF.
# More ConvNets in the 2020s: Scaling up Kernels Beyond 51 × 51 using Sparsity (https://arxiv.org/pdf/2207.03620.pdf)
# Github source: https://github.com/VITA-Group/SLaK
# Licensed under The MIT License [see LICENSE for details]
# Modified from https://github.com/DingXiaoH/RepLKNet-pytorch.
# --------------------------------------------------------'

from models.SLaK import SLaK

class SLakForERF(SLaK):

    def __init__(self,  kernel_size, width_factor=1, Decom=None
                 ):
        super().__init__(in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., kernel_size=kernel_size, width_factor=width_factor, Decom=Decom)
        # The default model here is SLaK-T. Changing dims for SLaK-S/B.
    def forward(self, x):
        x = self.forward_features(x)
        return x
        # return self.norm(x)     #   Using the feature maps after the final norm also makes sense. Observed very little difference.

