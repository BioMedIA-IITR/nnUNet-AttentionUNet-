from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.network_architecture.attention_unet import AttentionUNet
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class nnUNetTrainerAttentionUNet(nnUNetTrainer):
    """
    nnU-Net v2 trainer using Attention U-Net instead of the default U-Net.
    """

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        arch_kwargs = configuration_manager.network_arch_init_kwargs

        model = AttentionUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            n_stages=arch_kwargs["n_stages"],
            features_per_stage=arch_kwargs["features_per_stage"],
            kernel_sizes=arch_kwargs["kernel_sizes"],
            strides=arch_kwargs["strides"],
            n_conv_per_stage=arch_kwargs["n_conv_per_stage"],
            n_conv_per_stage_decoder=arch_kwargs["n_conv_per_stage_decoder"],
            conv_bias=arch_kwargs.get("conv_bias", True),
            deep_supervision=enable_deep_supervision,
        )

        return model
