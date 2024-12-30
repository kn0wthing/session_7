import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Input Block
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),      # input: 28x28x1 output: 28x28x8 RF: 3x3
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        
        # First Block
        self.block1 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),     # input: 14x14x8 output: 14x14x16 RF: 5x5
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Second Block
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),    # input: 7x7x16 output: 7x7x32 RF: 9x9
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Third Block
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),    # input: 7x7x32 output: 7x7x16 RF: 13x13
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Output Block
        self.output_block = nn.Sequential(
            nn.Conv2d(16, 10, 1)                # input: 7x7x16 output: 7x7x10 RF: 13x13
        )
        
        # Pooling layers
        self.pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.BatchNorm2d()
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(0.08)

    def forward(self, x):
        # Input block
        x = self.input_block(x)         # 28x28x8
        x = self.pool(x)                # 14x14x8
        
        # First block
        x = self.dropout(x)
        x = self.block1(x)              # 14x14x16
        x = self.pool(x)                # 7x7x16
        
        # Second block with dropout
        x = self.dropout(x)
        x = self.block2(x)              # 7x7x16
        
        # Third block with dropout
        x = self.dropout(x)
        x = self.block3(x)              # 7x7x16
        
        # Output block
        x = self.output_block(x)        # 7x7x10
        
        # Global average pooling and reshape
        x = F.adaptive_avg_pool2d(x, 1) # 1x1x10
        x = x.view(-1, 10)              # 10
        
        return F.log_softmax(x, dim=1)

def get_model():
    """
    Returns an instance of the model
    """
    return Net()

def get_summary(model):
    """
    Returns model summary including parameter count
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params
    }
