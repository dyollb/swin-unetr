import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet


class CascadedUNet(torch.nn.Module):
    """Cascaded network that uses feature channels as input to a UNet

    A list of feature networks in a first layer is concatenated with the input
    image. The feature channels are trained separately (possibly with more data),
    to make the CascadedUNet more robust.

    'feature_channel_list' lists the number of output channels of the feature
    networks. If 'include_background_channel' is False, the channel==0 is not forwarded
    to the second layer network.
    """

    def __init__(
        self,
        in_channels: int,
        feature_channel_list: list[int],
        out_channels: int,
        include_background_channel: bool = False,
    ):
        super().__init__()

        self.include_background_channel = include_background_channel
        self.feature_nets = torch.nn.ModuleList()
        sum_feature_channels = 0
        for feature_channels in feature_channel_list:
            if self.include_background_channel:
                sum_feature_channels += feature_channels
            else:
                sum_feature_channels += feature_channels - 1

            net = UNet(
                spatial_dims=3,
                in_channels=in_channels,
                # features + background
                out_channels=feature_channels,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                dropout=0.0,
                num_res_units=2,
                norm=Norm.BATCH,
            )
            self.feature_nets.append(net)
        self.feature_nets.requires_grad_(False)

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
        x_list = [m(x) for m in self.feature_nets]
        if not self.include_background_channel:
            # discard background (channel==0)
            x_list = [xi[:, 1:, ...] for xi in x_list]
        x_combined = torch.cat([x] + x_list, dim=1)
        return self.net(x_combined)
