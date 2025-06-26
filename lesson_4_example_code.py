# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 19:32:33 2025

@author: Gavin
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

# Set random seed for reproducibility
torch.manual_seed(42)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class MNIST_CAM_Attention_CNN(nn.Module):
    """MNIST CNN with parallel attention branch for CAM visualization"""
    
    def __init__(self, num_classes=10):
        super(MNIST_CAM_Attention_CNN, self).__init__()
        
        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28x32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x32
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14x64
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x64
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 7x7x64
        self.relu3 = nn.ReLU()
        
        # Attention branch - parallel to main classification branch
        # Global attention: 64 channels -> 1 channel for global spatial attention
        self.attention_conv = nn.Conv2d(64, 1, kernel_size=1)  # 1x1 conv: 64->1 channels
        self.attention_sigmoid = nn.Sigmoid()  # Sigmoid activation for attention maps
        
        # Main classification branch
        self.feature_conv = nn.Conv2d(64, 64, kernel_size=1)  # 1x1 conv for features
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 1x1x64
        
        # Classification head
        self.classifier = nn.Linear(64, num_classes)
        
        # Store intermediate results for visualization
        self.feature_maps = None
        self.attention_maps = None
        self.attended_features = None
        
    def forward(self, x):
        # Shared convolutional backbone
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu3(self.conv3(x))
        
        # Store the base feature maps
        base_features = x.clone()
        
        # Attention branch - generates single global attention map
        attention_features = self.attention_conv(x)  # 64 channels -> 1 channel: (batch, 1, 7, 7)
        attention_maps = self.attention_sigmoid(attention_features)  # Global attention map
        
        # Main feature branch
        feature_maps = self.feature_conv(x)  # Transform features for classification
        
        # Apply global attention to all feature channels (broadcasting)
        # attention_maps: (batch, 1, 7, 7) broadcasts to (batch, 64, 7, 7)
        attended_features = feature_maps * attention_maps
        
        # Store for visualization
        self.feature_maps = feature_maps.clone()
        self.attention_maps = attention_maps.clone()
        self.attended_features = attended_features.clone()
        
        # Global Average Pooling on attended features
        pooled_features = self.global_avg_pool(attended_features)  # Shape: (batch_size, 64, 1, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # Shape: (batch_size, 64)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output
    
    def get_cam(self, class_idx, input_size=(7, 7)):
        """Generate Class Activation Map for a specific class"""
        if self.attention_maps is None or self.feature_maps is None:
            raise ValueError("No feature maps available. Run forward pass first.")
        
        # Get the weights for the specified class
        classifier_weights = self.classifier.weight[class_idx]  # Shape: (64,)
        
        # Get the last batch's attention map (single channel) and feature maps
        batch_attention = self.attention_maps[-1, 0]  # Last sample, single channel: (7, 7)
        batch_features = self.feature_maps[-1]        # Last sample in batch: (64, 7, 7)
        
        # Generate CAM by weighted combination of features, modulated by global attention
        cam = torch.zeros(input_size, device=self.attention_maps.device)
        
        for i, weight in enumerate(classifier_weights):
            # Weight each feature channel and apply global attention
            channel_contribution = weight * batch_features[i] * batch_attention
            cam += channel_contribution
        
        # Apply ReLU to focus on positive contributions
        cam = torch.relu(cam)
        
        # Normalize CAM
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def get_attention_map(self):
        """Get global attention map for visualization"""
        if self.attention_maps is None:
            raise ValueError("No attention maps available. Run forward pass first.")
        
        # Return the single-channel global attention map
        attention = self.attention_maps[-1, 0]  # Last sample, single channel: (7, 7)
        return attention.detach().cpu().numpy()

# Create and train the model
model = MNIST_CAM_Attention_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model Architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

def train_model(num_epochs=10):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate epoch statistics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return train_losses, train_accuracies

def test_model():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    print(f'\nTest Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)')
    return accuracy

def visualize_cam_and_attention(model, dataset, num_samples=5):
    """Visualize CAM and attention maps for sample images"""
    model.eval()
    
    # Get some test samples
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    fig.suptitle('Original Image | Attention Map | CAM | Overlay', fontsize=16)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            predicted_class = torch.argmax(output, dim=1).item()
            
            # Get original image
            original_img = data[0, 0].cpu().numpy()
            
            # Get attention map (average across channels)
            attention_map = model.get_attention_map()
            
            # Get CAM for predicted class
            cam = model.get_cam(predicted_class)
            
            # Resize CAM and attention map to original image size
            cam_resized = cv2.resize(cam, (28, 28))
            attention_resized = cv2.resize(attention_map, (28, 28))
            
            # Create overlay
            overlay = original_img.copy()
            overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())  # Normalize
            
            # Plot results
            row = axes[i] if num_samples > 1 else axes
            
            # Original image
            row[0].imshow(original_img, cmap='gray')
            row[0].set_title(f'Original (True: {target.item()}, Pred: {predicted_class})')
            row[0].axis('off')
            
            # Attention map
            row[1].imshow(attention_resized, cmap='hot')
            row[1].set_title('Attention Map')
            row[1].axis('off')
            
            # CAM
            row[2].imshow(cam_resized, cmap='hot')
            row[2].set_title('CAM')
            row[2].axis('off')
            
            # Overlay
            row[3].imshow(overlay, cmap='gray', alpha=0.7)
            row[3].imshow(cam_resized, cmap='hot', alpha=0.3)
            row[3].set_title('CAM Overlay')
            row[3].axis('off')
    
    plt.tight_layout()
    plt.show()

# Train the model
print("\nStarting training...")
train_losses, train_accuracies = train_model(num_epochs=10)

# Test the model
print("\nTesting the model...")
test_accuracy = test_model()

# Visualize training progress
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot([acc * 100 for acc in train_accuracies], 'r-', label='Training Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize CAM and attention maps
print("\nGenerating CAM and attention visualizations...")
visualize_cam_and_attention(model, test_dataset, num_samples=5)