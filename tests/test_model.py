import sys
import os
import torch
import pytest

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MnistNet_1, MnistNet_2, MnistNet_3

# List of all model classes to test
MODEL_CLASSES = [MnistNet_1, MnistNet_2, MnistNet_3]

@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_creation(model_class):
    """Test if models can be instantiated"""
    model = model_class()
    assert isinstance(model, model_class)

@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_forward(model_class):
    """Test forward pass with single sample"""
    model = model_class()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)

@pytest.mark.parametrize("model_class", [MnistNet_1])
def test_parameter_count_for_mnistnet_1(model_class):
    """Test if model parameters are within limit"""
    model = model_class()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_class.__name__} Parameters: {total_params:,}")
    
    assert total_params < 10000, f"Model has {total_params:,} parameters, which exceeds the limit of 10000"

@pytest.mark.parametrize("model_class", [MnistNet_2, MnistNet_3])
def test_parameter_count_for_mnistnet_2_and_3(model_class):
    """Test if model parameters are within limit"""
    model = model_class()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_class.__name__} Parameters: {total_params:,}")
    
    assert total_params < 8000, f"Model has {total_params:,} parameters, which exceeds the limit of 8000"

@pytest.mark.parametrize("model_class", [MnistNet_2, MnistNet_3])
def test_batch_norm_layers(model_class):
    """Test presence of BatchNorm layers in models that should have them"""
    model = model_class()
    has_bn = False
    bn_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            has_bn = True
            bn_count += 1
    assert has_bn, f"{model_class.__name__} should include BatchNorm layers"
    print(f"{model_class.__name__} has {bn_count} BatchNorm layers")

@pytest.mark.parametrize("model_class", [MnistNet_2, MnistNet_3])
def test_dropout_layers(model_class):
    """Test presence and rate of Dropout layers"""
    model = model_class()
    has_dropout = False
    dropout_rates = []
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            has_dropout = True
            dropout_rates.append(module.p)
    
    assert has_dropout, f"{model_class.__name__} should include Dropout layers"
    
    # Test specific dropout rates
    expected_rate = 0.05 if model_class == MnistNet_2 else 0.01
    assert all(rate == expected_rate for rate in dropout_rates), \
        f"Expected dropout rate {expected_rate} in {model_class.__name__}"

@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_training_shape(model_class):
    """Test model output shape with batch processing"""
    model = model_class()
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), \
        f"Expected output shape {(batch_size, 10)}, got {output.shape}"

@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_output_range(model_class):
    """Test if model outputs valid probability distributions"""
    model = model_class()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    # Test if outputs are valid log probabilities
    assert torch.allclose(torch.exp(output).sum(), torch.tensor(1.0), atol=1e-5), \
        f"{model_class.__name__} outputs should sum to 1 after exp"
    assert (output <= 0).all(), f"{model_class.__name__} log_softmax outputs should be negative"

@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_convolution_layers(model_class):
    """Test presence and configuration of convolution layers"""
    model = model_class()
    conv_layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)
    
    assert len(conv_layers) > 0, f"{model_class.__name__} should include Convolution layers"
    
    # Test if all conv layers have bias=False as specified
    assert all(not layer.bias for layer in conv_layers), \
        f"{model_class.__name__} convolution layers should not have bias"

@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_on_device(model_class):
    """Test if model can be moved to available device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    x = torch.randn(1, 1, 28, 28).to(device)
    output = model(x)
    assert output.device == device, f"{model_class.__name__} output should be on {device}"

@pytest.mark.parametrize("model_class", [MnistNet_2, MnistNet_3])
def test_global_average_pooling(model_class):
    """Test presence of Global Average Pooling in appropriate models"""
    model = model_class()
    has_gap = False
    for module in model.modules():
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            has_gap = True
            break
    assert has_gap, f"{model_class.__name__} should include Global Average Pooling"

def test_model_architectural_differences():
    """Test key architectural differences between models"""
    model1 = MnistNet_1()
    model2 = MnistNet_2()
    model3 = MnistNet_3()
    
    # Test that MnistNet_1 doesn't have BatchNorm
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model1.modules())
    assert not has_bn, "MnistNet_1 should not have BatchNorm layers"
    
    # Test that MnistNet_2 has higher dropout rate than MnistNet_3
    dropout2 = next(m.p for m in model2.modules() if isinstance(m, torch.nn.Dropout))
    dropout3 = next(m.p for m in model3.modules() if isinstance(m, torch.nn.Dropout))
    assert dropout2 > dropout3, "MnistNet_2 should have higher dropout rate than MnistNet_3" 