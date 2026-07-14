import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMNISTConvNet(nn.Module):
    def __init__(self):
        super(SimpleMNISTConvNet, self).__init__()
        
        # 1. Conv-Block (Output: 32 Kanäle)
        self.Conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.Pool1 = nn.MaxPool2d(2, 2)
        
        # 2. Conv-Block (Output: 64 Kanäle) -> Hier lag der Unterschied!
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.Pool2 = nn.MaxPool2d(2, 2)
        
        # 3. Conv-Block (Output: 64 Kanäle)
        self.Conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Linear-Layer: 64 Kanäle * 7 * 7 = 3136 Features (genau wie im Original)
        self.FC1 = nn.Linear(3136, 10)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv1 -> ReLU -> MaxPool1: (1, 28, 28) -> (32, 28, 28) -> (32, 14, 14)
        x = self.Pool1(F.relu(self.Conv1(x)))
        
        # Conv2 -> ReLU -> MaxPool2: (32, 14, 14) -> (64, 14, 14) -> (64, 7, 7)
        x = self.Pool2(F.relu(self.Conv2(x)))
        
        # Conv3 -> ReLU: (64, 7, 7) -> (64, 7, 7)
        x = F.relu(self.Conv3(x))
        
        # Flatten: (64, 7, 7) -> 3136 Features
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = self.FC1(x)
        return x
    
# Load model
def load_simple_mnist_model(model_path, model, optimizer=None, device='cpu'):
    """Loads a saved model"""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    accuracy = checkpoint['test_accuracy']
    
    print(f"Model loaded - Epoch: {epoch}, Accuracy: {accuracy:.2f}%")
    return model, optimizer, epoch, accuracy