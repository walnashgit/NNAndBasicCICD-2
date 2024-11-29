import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.mnist_model import MNISTNet
from datetime import datetime
import torch.nn.functional as F
import os
from tqdm import tqdm
from torchsummary import summary
import glob

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def delete_old_models():
    """Delete all existing model files"""
    model_files = glob.glob('models/mnist_model_*.pth')
    for f in model_files:
        try:
            os.remove(f)
            print(f"Deleted old model: {f}")
        except OSError as e:
            print(f"Error deleting {f}: {e}")

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{test_loss/total:.4f}',
                'accuracy': f'{accuracy:.2f}%'
            })
    
    test_loss /= total
    test_accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_accuracy:.2f}%)')
    return test_accuracy

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model initialization
    model = MNISTNet()
    
    # Print model summary before moving to device
    print("\nModel Summary:")
    if device.type == 'cuda' or device.type == 'mps':
        summary(model.to('cpu'), input_size=(1, 28, 28))
        model = model.to(device)
    else:
        summary(model, input_size=(1, 28, 28))
    
    # Print model parameters
    total_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Training
    num_epochs = 20
    best_accuracy = 0.0
    best_model_path = None
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            running_loss += loss.item()
            
            accuracy = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'accuracy': f'{accuracy:.2f}%'
            })
        
        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs} - Training Accuracy: {train_accuracy:.2f}%')
        
        # Testing phase
        print(f'\nEvaluating on test set...')
        test_accuracy = test(model, device, test_loader)
        
        # Save best model based on test accuracy
        if test_accuracy > best_accuracy:
            # Delete previous best model if it exists
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f'Deleted previous best model: {best_model_path}')
            
            best_accuracy = test_accuracy
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            accuracy_str = f"{test_accuracy:.1f}"
            best_model_path = f'models/mnist_model_{timestamp}_acc{accuracy_str}.pth'
            
            # Delete any other existing models
            delete_old_models()
            
            # Save new best model
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved to {best_model_path}')

    print(f'\nTraining completed. Best test accuracy: {best_accuracy:.2f}%')
    print(f'Best model saved at: {best_model_path}')

if __name__ == '__main__':
    train() 