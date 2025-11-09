import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMNISTConvNet(nn.Module):
    """
    Convolutional Neural Network - better for image processing
    
    Architecture:
    Conv1 -> ReLU -> MaxPool -> Conv2 -> ReLU -> MaxPool -> Flatten -> FC1 -> ReLU -> FC2
    """
    
    def __init__(self):
        super(SimpleMNISTConvNet, self).__init__()
        
        # Convolutional Layer 1: 1 input channel (grayscale), 32 output channels
        # Kernel size 3x3, padding 1 (maintains image size)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # Convolutional Layer 2: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # MaxPooling: Reduces image size by factor of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 2x pooling: 28x28 -> 14x14 -> 7x7
        # 64 channels * 7 * 7 = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout: Randomly turns off 50% of neurons (prevents overfitting)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv1 -> ReLU -> MaxPool: (1, 28, 28) -> (32, 28, 28) -> (32, 14, 14)
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv2 -> ReLU -> MaxPool: (32, 14, 14) -> (64, 14, 14) -> (64, 7, 7)
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten: (64, 7, 7) -> (3136)
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully Connected Layer 1 with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output Layer
        x = self.fc2(x)
        
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