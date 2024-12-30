"""
This module contains three different CNN architectures for MNIST digit classification.
Each model progressively introduces more advanced techniques:
- MnistNet_1: Basic CNN with conv layers and max pooling
- MnistNet_2: Adds BatchNorm and Dropout (0.05)
- MnistNet_3: Refined architecture with lower dropout (0.01) and additional BatchNorm
"""

import torch.nn as nn
import torch.nn.functional as F


class MnistNet_1(nn.Module):
    """
    Basic CNN architecture for MNIST classification
    Input: 28x28x1 image
    Output: 10 classes (digits 0-9)
    """
    def __init__(self):
        super(MnistNet_1, self).__init__()

        # Conv Block 1: Input: 28x28 -> Output: 11x11
        self.conv_block_1 = nn.Sequential(
            # RF: 3x3, Input: 28x28 -> Output: 26x26
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False),
            nn.ReLU(),

            # RF: 5x5, Input: 26x26 -> Output: 24x24
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),
            nn.ReLU(),

            # RF: 7x7, Input: 24x24 -> Output: 22x22
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False),
            nn.ReLU(),

            # RF: 7x7, Input: 22x22 -> Output: 22x22
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False),
            # RF: 8x8, Input: 22x22 -> Output: 11x11
            nn.MaxPool2d(kernel_size=2, stride=2)
        )   

        # Conv Block 2: Input: 11x11 -> Output: 5x5
        self.conv_block_2 = nn.Sequential(
            # RF: 10x10, Input: 11x11 -> Output: 9x9
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False),
            nn.ReLU(),

            # RF: 12x12, Input: 9x9 -> Output: 7x7
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, bias=False),
            nn.ReLU(),

            # RF: 14x14, Input: 7x7 -> Output: 5x5
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),
            nn.ReLU(),

            # RF: 14x14, Input: 5x5 -> Output: 5x5
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False),
            nn.ReLU(),
        )

        # Conv Block 3: Input: 5x5 -> Output: 1x1
        self.conv_block_3 = nn.Sequential(
            # RF: 18x18, Input: 5x5 -> Output: 1x1
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, bias=False)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = x.view(-1, 10)  # Flatten the tensor
        return F.log_softmax(x, dim=-1)


class MnistNet_2(nn.Module):
    """
    Enhanced CNN architecture with BatchNorm and Dropout
    Input: 28x28x1 image
    Output: 10 classes (digits 0-9)
    Dropout rate: 0.05
    """
    def __init__(self):
        super(MnistNet_2, self).__init__()
        drop_rate = 0.05

        # Conv Block 1: Input: 28x28 -> Output: 12x12
        self.conv_block_1 = nn.Sequential(
            # RF: 3x3, Input: 28x28 -> Output: 26x26
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(drop_rate),

            # RF: 5x5, Input: 26x26 -> Output: 24x24
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),

            # RF: 5x5, Input: 24x24 -> Output: 24x24
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False),
            # RF: 6x6, Input: 24x24 -> Output: 12x12
            nn.MaxPool2d(kernel_size=2, stride=2)
        )   

        # Conv Block 2: Input: 12x12 -> Output: 6x6
        self.conv_block_2 = nn.Sequential(
            # RF: 8x8, Input: 12x12 -> Output: 10x10
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),

            # RF: 10x10, Input: 10x10 -> Output: 8x8
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),

            # RF: 12x12, Input: 8x8 -> Output: 6x6
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),
        )
        
        # Global Average Pooling: Input: 6x6 -> Output: 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Final Conv: Input: 1x1 -> Output: 1x1
        self.conv_block_3 = nn.Sequential(
            # RF: 12x12, Input: 1x1 -> Output: 1x1
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.avg_pool(x)
        x = self.conv_block_3(x)
        x = x.view(-1, 10)  # Flatten the tensor
        return F.log_softmax(x, dim=-1)


class MnistNet_3(nn.Module):
    """
    Refined CNN architecture with lower dropout and additional BatchNorm
    Input: 28x28x1 image
    Output: 10 classes (digits 0-9)
    Dropout rate: 0.01
    """
    def __init__(self):
        super(MnistNet_3, self).__init__()
        drop_rate = 0.01

        # Conv Block 1: Input: 28x28 -> Output: 12x12
        self.conv_block_1 = nn.Sequential(
            # RF: 3x3, Input: 28x28 -> Output: 26x26
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(drop_rate),

            # RF: 5x5, Input: 26x26 -> Output: 24x24
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),

            # RF: 5x5, Input: 24x24 -> Output: 24x24
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            # RF: 6x6, Input: 24x24 -> Output: 12x12
            nn.MaxPool2d(kernel_size=2, stride=2)
        )   

        # Conv Block 2: Input: 12x12 -> Output: 6x6
        self.conv_block_2 = nn.Sequential(
            # RF: 8x8, Input: 12x12 -> Output: 10x10
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(drop_rate),

            # RF: 10x10, Input: 10x10 -> Output: 8x8
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),

            # RF: 12x12, Input: 8x8 -> Output: 6x6
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop_rate),
        )
        
        # Global Average Pooling: Input: 6x6 -> Output: 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Final Conv: Input: 1x1 -> Output: 1x1
        self.conv_block_3 = nn.Sequential(
            # RF: 12x12, Input: 1x1 -> Output: 1x1
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.avg_pool(x)
        x = self.conv_block_3(x)
        x = x.view(-1, 10)  # Flatten the tensor
        return F.log_softmax(x, dim=-1)