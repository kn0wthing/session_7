import torch
import torch.nn as nn

def count_parameters(model):
    """Count the total number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_batchnorm(model):
    """Check if model contains batch normalization layers"""
    return any(isinstance(module, nn.BatchNorm2d) for module in model.modules())

def check_dropout(model):
    """Check if model contains dropout layers"""
    return any(isinstance(module, nn.Dropout) for module in model.modules())

def check_linear(model):
    """Check if model contains fully connected layers"""
    return any(isinstance(module, nn.Linear) for module in model.modules()) 