import torch.nn as nn
from src.models.base_model import BaseModel
from src.utils.activation_factory import ActivationFactory
from src.utils.weight_init_factory import WeightInitFactory
import torch
from typing import Dict, Any
from torch import Tensor


class CNN(BaseModel):
    def __init__(
        self, params: Dict[str, Any], num_classes: int, input_size: int
    ) -> None:
        super(CNN, self).__init__()

        self.params = params
        self.num_classes = num_classes
        self.input_size = input_size
        self.conv_layers = self._build_conv_layers()
        self.fc_layers = self._build_fc_layers()
        self.flatten = nn.Flatten()

        self.initialize_weights()

    def _build_conv_layers(self) -> nn.Sequential:
        layers = []
        in_channels = 1

        dropout_enabled = self.params["dropout_enabled"]
        dropout_position = self.params["dropout_position"]
        dropout_rate = self.params["dropout_rate"]

        for i, out_channels in enumerate(self.params["conv_layers"]):
            block = []

            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.params["kernel_size"],
                stride=self.params["stride"],
                padding=self.params["kernel_size"] // 2,
            )
            block.append(conv)

            if i < len(self.params["conv_layers"]) - 1:
                block.append(nn.BatchNorm2d(out_channels))
            if self.params["use_group_norm"]:
                num_groups = min(4, out_channels) if out_channels % 4 == 0 else 1
                block.append(nn.GroupNorm(num_groups, out_channels))
            if self.params["use_instance_norm"]:
                block.append(nn.InstanceNorm2d(out_channels))

            if dropout_enabled and dropout_position == "after_conv":
                block.append(nn.Dropout2d(p=dropout_rate))

            block.append(
                ActivationFactory.get_activation(self.params["activation_conv"])
            )

            if dropout_enabled and dropout_position == "after_activation":
                block.append(nn.Dropout2d(p=dropout_rate))

            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            layers.extend(block)
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _build_fc_layers(self) -> nn.Sequential:
        layers = []

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_size, self.input_size)
            dummy_output: Tensor = self.conv_layers(dummy_input)
            in_features = dummy_output.view(1, -1).size(1)

        dropout_enabled = self.params["dropout_enabled"]
        dropout_position = self.params["dropout_position"]
        dropout_rate = self.params["dropout_rate"]

        for out_features in self.params["fully_connected_layers"]:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(
                ActivationFactory.get_activation(self.params["activation_fc"])
            )
            if dropout_enabled and dropout_position == "after_fc":
                layers.append(nn.Dropout(p=dropout_rate))
            in_features = out_features

        layers.append(nn.Linear(in_features, self.num_classes))

        return nn.Sequential(*layers)

    def initialize_weights(self) -> None:
        initializer = WeightInitFactory.get_initializer(
            self.params["weight_initialization"]
        )
        self.apply(initializer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.params["use_min_max_scaling"]:
            x = torch.stack(
                [
                    (
                        (xi - xi.min()) / (xi.max() - xi.min() + 1e-8)
                        if xi.max() > xi.min()
                        else xi
                    )
                    for xi in x
                ]
            )
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
