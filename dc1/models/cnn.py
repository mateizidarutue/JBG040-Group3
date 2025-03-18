import torch.nn as nn
from models.base_model import BaseModel
from utils.activation_factory import ActivationFactory
from utils.weight_init_factory import WeightInitFactory
import torch
from typing import Dict, Any


class CNN(BaseModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(CNN, self).__init__()

        self.config = config
        self.conv_layers = self._build_conv_layers()
        self.fc_layers = self._build_fc_layers()
        self.flatten = nn.Flatten()

        self.initialize_weights()

        if self.config["freeze_layers"]:
            self.freeze_conv_layers()

    def _build_conv_layers(self) -> nn.Sequential:
        layers = []
        in_channels = 1

        dropout_enabled = self.config["dropout_enabled"]
        dropout_position = self.config["dropout_position"]
        dropout_rate = self.config["dropout_rate"]

        for out_channels in self.config["conv_layers"]:
            block = []

            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.config["kernel_size"],
                stride=self.config["stride"],
                padding=self.config["padding"],
            )
            block.append(conv)

            if self.config["use_batch_norm"]:
                block.append(nn.BatchNorm2d(out_channels))
            if self.config["use_group_norm"]:
                block.append(nn.GroupNorm(4, out_channels))
            if self.config["use_instance_norm"]:
                block.append(nn.InstanceNorm2d(out_channels))

            if dropout_enabled and dropout_position == "after_conv":
                block.append(nn.Dropout2d(p=dropout_rate))

            block.append(
                ActivationFactory.get_activation(self.config["activation_conv"])
            )

            if dropout_enabled and dropout_position == "after_activation":
                block.append(nn.Dropout2d(p=dropout_rate))

            block.append(nn.MaxPool2d(kernel_size=2))

            layers.append(nn.Sequential(*block))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _build_fc_layers(self) -> nn.Sequential:
        layers = []
        num_pools = len(self.config["conv_layers"])
        conv_output_size = 128 // (2**num_pools)
        in_features = (
            conv_output_size * conv_output_size * self.config["conv_layers"][-1]
        )

        dropout_enabled = self.config["dropout_enabled"]
        dropout_position = self.config["dropout_position"]
        dropout_rate = self.config["dropout_rate"]

        for out_features in self.config["fully_connected_layers"]:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(
                ActivationFactory.get_activation(self.config["activation_fc"])
            )
            if dropout_enabled and dropout_position == "after_fc":
                layers.append(nn.Dropout(p=dropout_rate))
            in_features = out_features

        layers.append(nn.Linear(in_features, 6))
        return nn.Sequential(*layers)

    def initialize_weights(self) -> None:
        initializer = WeightInitFactory.get_initializer(
            self.config["weight_initialization"]
        )
        self.apply(initializer)

    def freeze_conv_layers(self) -> None:
        for module in self.conv_layers.modules():
            if isinstance(module, nn.Conv2d):
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config["use_min_max_scaling"]:
            x = torch.stack(
                [(xi - xi.min()) / (xi.max() - xi.min() + 1e-8) for xi in x]
            )
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
