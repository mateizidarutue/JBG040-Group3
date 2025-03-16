import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_classes: int = 6) -> None:
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            #Convolution Block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Convolution Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, groups=32),  # Depthwise Conv
            nn.Conv2d(64, 64, kernel_size=1),  # Pointwise Conv (Reduces parameters)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Convolution Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, groups=64),  # Depthwise Conv
            nn.Conv2d(128, 128, kernel_size=1),  # Pointwise Conv
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((4, 4))

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(144, 256),
            nn.Linear(256, n_classes)
        )

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        x = self.fc_layers(x)
        return x
