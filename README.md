# MNIST Classification Models

This repository contains three PyTorch CNN models for MNIST digit classification, each with different architectures and training approaches.

## Targets & Results

### MnistNet_1
**Target:**
- Parameters close to 8K
- >98% test accuracy within 10 epochs
- No batch normalization or regularization

**Result:**
- Parameters: 8,620
- Test Accuracy: 98.49%
- Training Accuracy: 98.67%

### MnistNet_2
**Target:**
- Parameters < 8K
- Implement batch normalization and dropout
- Experiment with dropout rates

**Result:**
- Parameters: 7,416
- Test Accuracy: 99.39%
- Training Accuracy: 98.82%


### MnistNet_3
**Target:**
- Parameters < 7K
- Optimize learning rate and scheduler

**Result:**
- Parameters: 6,560
- Test Accuracy: 99.45%
- Training Accuracy: 98.90%
- Achieved 99.4% in epoch 8

## Analysis

### MnistNet_1
- Used basic CNN architecture without regularization
- Achieved good accuracy through:
  - Strategic placement of MaxPooling
  - Balanced channel sizes (8->16->8)
  - 1x1 convolutions for channel reduction
- Higher parameter count due to lack of regularization techniques

### MnistNet_2
- Improved accuracy through:
  - BatchNorm after each conv layer
  - Moderate dropout (0.05) for regularization
  - Global Average Pooling to reduce parameters
- Better generalization despite fewer parameters
- Balanced trade-off between model size and accuracy

### MnistNet_3
- Best performance through:
  - Lower dropout rate (0.01) to retain more information
  - Optimized learning rate schedule (step_size=4, gamma=0.35)
  - Additional BatchNorm layers
  - Efficient channel progression (8->12->16)

## Network Architecture

### Layer Configuration

| Layer Type | MnistNet_1 | MnistNet_2 | MnistNet_3 |
|------------|------------|------------|------------|
| Input | 28x28x1 | 28x28x1 | 28x28x1 |
| Conv Block 1 | 3x3 (8) -> 3x3 (16) -> 3x3 (16) -> 1x1 (8) | 3x3 (8) -> BN -> Drop -> 3x3 (16) -> BN -> Drop | 3x3 (8) -> BN -> Drop -> 3x3 (16) -> BN -> Drop |
| MaxPool | 2x2 | 2x2 | 2x2 |
| Conv Block 2 | 3x3 (8) -> 3x3 (8) -> 3x3 (16) -> 1x1 (10) | 3x3 (16) -> BN -> Drop -> 3x3 (16) -> BN -> Drop | 3x3 (12) -> BN -> Drop -> 3x3 (16) -> BN -> Drop |
| Final Block | 5x5 (10) | GAP -> 1x1 (10) | GAP -> 1x1 (10) |
| Output | 10 classes | 10 classes | 10 classes |

## Data Transformations

### MnistNet_1
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### MnistNet_2 & MnistNet_3
```python
transforms.Compose([
    transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
    transforms.RandomAffine(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## Training Configuration

### MnistNet_1
- Optimizer: SGD(lr=0.01, momentum=0.9)
- Loss: CrossEntropyLoss
- No scheduler

### MnistNet_2
- Optimizer: SGD(lr=0.075, momentum=0.9, nesterov=True)
- Loss: NLLLoss
- Scheduler: StepLR(step_size=2, gamma=0.65)
- Dropout: 0.05

### MnistNet_3
- Optimizer: SGD(lr=0.05, momentum=0.9)
- Loss: NLLLoss
- Scheduler: StepLR(step_size=4, gamma=0.35)
- Dropout: 0.01

## Requirements
- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- tqdm

## Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mnist-classification.git
cd mnist-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train all models:
```bash
python src/train_all.py
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
