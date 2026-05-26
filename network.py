from SSAN import ssan
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, final_sigmoid=True):
        super(Network, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        self.Generator = ssan(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid
        )

    def forward(self, x):
        fake_x = self.Generator(x)
        return fake_x
