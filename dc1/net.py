import torch
import torch.nn as nn

class MinMaxNormalization(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        """
        Min-Max Normalization Layer
        :param min_val: Minimum value of the scaled range (default 0)
        :param max_val: Maximum value of the scaled range (default 1)
        """
        super(MinMaxNormalization, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        """
        Normalize input x to a specified range [min_val, max_val]
        """
        min_x = x.amin(dim=(1, 2, 3), keepdim=True)  # Find min per sample
        max_x = x.amax(dim=(1, 2, 3), keepdim=True)  # Find max per sample
        
        x = (x - min_x) / (max_x - min_x + 1e-8)  # Normalize to [0,1]
        x = x * (self.max_val - self.min_val) + self.min_val  # Scale to [min_val, max_val]

        return x

class Net(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net, self).__init__()

        self.min_max_norm = MinMaxNormalization()  # Min-Max Normalization Layer

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
            torch.nn.Dropout(p=0.5, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            torch.nn.Dropout(p=0.25, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 16, kernel_size=4, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.125, inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(144, 256),
            nn.Linear(256, n_classes)
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.min_max_norm(x)  # Apply Min-Max Normalization
        x = self.cnn_layers(x)
        # Flatten the input for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
