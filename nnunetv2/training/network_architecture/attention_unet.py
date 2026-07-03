from typing import List, Sequence

import torch
from torch import nn


def get_conv(dim: int):
    return nn.Conv2d if dim == 2 else nn.Conv3d


def get_conv_transpose(dim: int):
    return nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d


def get_norm(dim: int):
    return nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        kernel_size: Sequence[int],
        n_convs: int,
        conv_bias: bool = True,
    ):
        super().__init__()

        Conv = get_conv(dim)
        Norm = get_norm(dim)

        layers = []
        for i in range(n_convs):
            ic = in_channels if i == 0 else out_channels
            padding = tuple(k // 2 for k in kernel_size)

            layers.append(
                Conv(
                    ic,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=conv_bias,
                )
            )
            layers.append(Norm(out_channels, eps=1e-5, affine=True))
            layers.append(nn.LeakyReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int, dim: int):
        super().__init__()

        Conv = get_conv(dim)
        Norm = get_norm(dim)

        self.theta_x = nn.Sequential(
            Conv(skip_channels, inter_channels, kernel_size=1, bias=True),
            Norm(inter_channels, eps=1e-5, affine=True),
        )

        self.phi_g = nn.Sequential(
            Conv(gate_channels, inter_channels, kernel_size=1, bias=True),
            Norm(inter_channels, eps=1e-5, affine=True),
        )

        self.psi = nn.Sequential(
            Conv(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, skip, gate):
        attention = self.relu(self.theta_x(skip) + self.phi_g(gate))
        attention = self.psi(attention)
        return skip * attention


class AttentionUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: List[int],
        kernel_sizes: List[Sequence[int]],
        strides: List[Sequence[int]],
        n_conv_per_stage: List[int],
        n_conv_per_stage_decoder: List[int],
        conv_bias: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.n_stages = n_stages
        self.dim = len(kernel_sizes[0])

        Conv = get_conv(self.dim)
        ConvTranspose = get_conv_transpose(self.dim)

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_channels = input_channels

        for s in range(n_stages):
            self.encoders.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=features_per_stage[s],
                    dim=self.dim,
                    kernel_size=kernel_sizes[s],
                    n_convs=n_conv_per_stage[s],
                    conv_bias=conv_bias,
                )
            )

            if s < n_stages - 1:
                self.downsamples.append(
                    Conv(
                        features_per_stage[s],
                        features_per_stage[s],
                        kernel_size=strides[s + 1],
                        stride=strides[s + 1],
                        bias=conv_bias,
                    )
                )

            in_channels = features_per_stage[s]

        self.upsamples = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        for s in range(n_stages - 2, -1, -1):
            self.upsamples.append(
                ConvTranspose(
                    features_per_stage[s + 1],
                    features_per_stage[s],
                    kernel_size=strides[s + 1],
                    stride=strides[s + 1],
                    bias=conv_bias,
                )
            )

            self.attention_gates.append(
                AttentionGate(
                    skip_channels=features_per_stage[s],
                    gate_channels=features_per_stage[s],
                    inter_channels=max(features_per_stage[s] // 2, 1),
                    dim=self.dim,
                )
            )

            self.decoders.append(
                ConvBlock(
                    in_channels=features_per_stage[s] * 2,
                    out_channels=features_per_stage[s],
                    dim=self.dim,
                    kernel_size=kernel_sizes[s],
                    n_convs=n_conv_per_stage_decoder[n_stages - 2 - s],
                    conv_bias=conv_bias,
                )
            )

            self.seg_layers.append(
                Conv(
                    features_per_stage[s],
                    num_classes,
                    kernel_size=1,
                    bias=True,
                )
            )

    def forward(self, x):
        skips = []

        for s in range(self.n_stages):
            x = self.encoders[s](x)
            skips.append(x)

            if s < self.n_stages - 1:
                x = self.downsamples[s](x)

        seg_outputs = []

        for decoder_idx in range(self.n_stages - 1):
            skip_idx = self.n_stages - 2 - decoder_idx

            x = self.upsamples[decoder_idx](x)

            skip = skips[skip_idx]
            skip = self.attention_gates[decoder_idx](skip, x)

            x = torch.cat((skip, x), dim=1)
            x = self.decoders[decoder_idx](x)

            seg_outputs.append(self.seg_layers[decoder_idx](x))

        if self.deep_supervision:
            return seg_outputs

        return seg_outputs[0]

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
