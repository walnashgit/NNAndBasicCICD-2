import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model.mnist_model import MNISTNet
import pytest
import glob
import os

def get_latest_model():
    model_files = glob.glob('models/mnist_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model files found")
    return max(model_files)

def test_model_architecture():
    model = MNISTNet()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape is incorrect"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be < 25000"

def test_batch_normalization():
    model = MNISTNet()
    has_batchnorm = False
    
    # Check if model contains batch normalization layers
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            has_batchnorm = True
            break
    
    assert has_batchnorm, "Model should use Batch Normalization"

def test_dropout():
    model = MNISTNet()
    has_dropout = False
    
    # Check if model contains dropout layers
    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
            has_dropout = True
            break
    
    assert has_dropout, "Model should use Dropout"

def test_gap_or_fc():
    model = MNISTNet()
    has_gap = False
    has_fc = False
    
    # Check for Global Average Pooling or Fully Connected layers
    for module in model.modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            has_gap = True
        elif isinstance(module, nn.Linear):
            has_fc = True
    
    assert has_gap or has_fc, "Model should use either Global Average Pooling or Fully Connected layer"

def test_parameter_count():
    model = MNISTNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    assert total_params < 20000, f"Model has too many parameters ({total_params})"

def test_model_accuracy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    
    # Load the latest model
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be > 95%"

if __name__ == "__main__":
    pytest.main([__file__]) 