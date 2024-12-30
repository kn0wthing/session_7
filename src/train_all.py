from __future__ import print_function
import random
import os
import ssl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchinfo import summary
from tqdm import tqdm
from model import MnistNet_1, MnistNet_2, MnistNet_3

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

def seed_everything(seed=1):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_data_loaders(batch_size, kwargs, augment=False):
    """Create train and test data loaders"""
    # Base transforms
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    
    # Augmentation transforms
    aug_transforms = [
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.RandomAffine(degrees=15),
    ] if augment else []
    
    train_transforms = transforms.Compose(aug_transforms + base_transforms)
    test_transforms = transforms.Compose(base_transforms)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=train_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=test_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader

def train(model, device, train_loader, optimizer, criterion, epoch):
    """Training function"""
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'Epoch={epoch} Batch={batch_idx} loss={loss.item():.7f} Accuracy={100. * correct / len(train_loader.dataset):.2f}%')

def test(model, device, test_loader, criterion):
    """Testing function"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.7f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train_model_1(device, train_loader, test_loader):
    """Training configuration for MnistNet_1"""
    print("\n=== Training MnistNet_1 ===")
    model = MnistNet_1().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(15):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

def train_model_2(device, train_loader, test_loader):
    """Training configuration for MnistNet_2"""
    print("\n=== Training MnistNet_2 ===")
    model = MnistNet_2().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.075, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.65, verbose=True)
    criterion = nn.NLLLoss()
    
    for epoch in range(15):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)
        scheduler.step()

def train_model_3(device, train_loader, test_loader):
    """Training configuration for MnistNet_3"""
    print("\n=== Training MnistNet_3 ===")
    model = MnistNet_3().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.35, verbose=True)
    criterion = nn.NLLLoss()
    
    for epoch in range(15):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)
        scheduler.step()

def main():
    # Setup
    seed_everything(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    batch_size = 128

    # Get data loaders
    train_loader_basic, test_loader = get_data_loaders(batch_size, kwargs, augment=False)
    train_loader_aug, _ = get_data_loaders(batch_size, kwargs, augment=True)

    # Train MnistNet_1 (without augmentation)
    train_model_1(device, train_loader_basic, test_loader)
    
    # Train MnistNet_2 (with augmentation)
    train_model_2(device, train_loader_aug, test_loader)
    
    # Train MnistNet_3 (with augmentation)
    train_model_3(device, train_loader_aug, test_loader)

if __name__ == '__main__':
    main() 