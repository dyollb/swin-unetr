from typing import List

import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet


class CascadedUNet(torch.nn.Module):
    """Cascaded network that uses feature channels as input to a UNet

    A list of feature networks in a first layer is concatenated with the input
    image. The feature channels are trained separately (possibly with more data),
    to make the CascadedUNet more robust.
    """

    def __init__(
        self,
        in_channels: int,
        feature_channel_list: List[int],
        out_channels: int,
    ):
        super().__init__()

        self.feature_nets: List[UNet] = []
        sum_feature_channels = 0
        for feature_channels in feature_channel_list:
            sum_feature_channels += feature_channels
            net = UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=feature_channels,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                dropout=0.0,
                num_res_units=2,
                norm=Norm.BATCH,
            )
            net.requires_grad_(False)
            self.feature_nets.append(net)

        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels + sum_feature_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.0,
            num_res_units=2,
            norm=Norm.BATCH,
        )

    def forward(self, x):
        x_list = [x] + [m(x) for m in self.feature_nets]
        x_combined = torch.cat(x_list)
        return self.net(x_combined)

    def string(self):
        r = "\n".join([f"Feature {i}: {n}" for i, n in enumerate(self.feature_nets)])
        return r + "\nNetwork: " + {self.net}
