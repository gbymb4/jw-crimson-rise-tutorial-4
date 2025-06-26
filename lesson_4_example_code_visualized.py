# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 19:40:12 2025

@author: Gavin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ImprovedHighResCNN(nn.Module):
    """
    Improved CNN with better spatial attention for high-quality CAM visualization
    Key improvements:
    1. More gradual channel reduction
    2. Larger kernels in final layers for better spatial context
    3. Skip connections for better gradient flow
    4. Spatial attention mechanism
    """
    def __init__(self, num_classes=10):
        super(ImprovedHighResCNN, self).__init__()
        
        # Early feature extraction with small kernels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Dilated convolutions with medium receptive field
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Larger kernels for spatial context (key improvement!)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # 5x5 kernel
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5, padding=2)  # 5x5 kernel
        self.bn6 = nn.BatchNorm2d(128)
        
        # Large kernel attention layer (major improvement!)
        self.attention_conv = nn.Conv2d(128, 256, kernel_size=7, padding=3)  # 7x7 kernel
        self.attention_bn = nn.BatchNorm2d(256)
        
        # Final feature maps with more channels for better discrimination
        self.final_conv = nn.Conv2d(256, 128, kernel_size=1)  # 1x1 to reduce channels from 256+256=512 combined
        self.final_bn = nn.BatchNorm2d(128)
        
        # Skip connection pathway
        self.skip_conv = nn.Conv2d(64, 256, kernel_size=1)  # Match channels for skip (256 to match attention_conv output)
        
        # Spatial attention module
        self.spatial_attention = SpatialAttentionModule(128)
        
        # Global Average Pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Early features
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        
        # Dilated features
        x2 = F.relu(self.bn3(self.conv3(x1)))
        x2 = F.relu(self.bn4(self.conv4(x2)))
        
        # Large kernel features
        x3 = F.relu(self.bn5(self.conv5(x2)))
        x3 = F.relu(self.bn6(self.conv6(x3)))
        
        # Attention features with large kernel
        x4 = F.relu(self.attention_bn(self.attention_conv(x3)))
        
        # Skip connection from x2 to help preserve spatial details
        skip = self.skip_conv(x2)  # Now 64 -> 256 channels
        
        # Combine features (both are now 256 channels)
        combined = x4 + skip
        
        # Final feature maps
        features = F.relu(self.final_bn(self.final_conv(combined)))  # 256 -> 128 channels
        
        # Apply spatial attention
        features = self.spatial_attention(features)
        
        # Global average pooling
        pooled = self.global_avg_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classifier
        output = self.fc(self.dropout(pooled))
        
        return output, features

class SpatialAttentionModule(nn.Module):
    """Spatial attention to enhance important regions"""
    def __init__(self, channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate spatial attention map
        attention = F.relu(self.conv1(x))
        attention = self.sigmoid(self.conv2(attention))
        
        # Apply attention
        return x * attention

class EnhancedCAMVisualizer:
    """Enhanced visualizer with multiple CAM techniques"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Register hooks for gradient-based methods
        self.gradients = []
        self.activations = []
        self.register_hooks()
    
    def register_hooks(self):
        """Register hooks for gradient collection"""
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])
            
        def forward_hook(module, input, output):
            self.activations.append(output)
        
        # Register on the final feature layer
        target_layer = self.model.final_conv
        target_layer.register_backward_hook(backward_hook)
        target_layer.register_forward_hook(forward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None, method='standard'):
        """Generate high-resolution CAM with multiple methods"""
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        if method == 'grad_cam':
            return self._generate_grad_cam(input_tensor, class_idx)
        else:
            return self._generate_standard_cam(input_tensor, class_idx)
    
    def _generate_standard_cam(self, input_tensor, class_idx=None):
        """Standard CAM generation"""
        with torch.no_grad():
            output, features = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1)
            
            batch_size = features.size(0)
            device = features.device
            cam = torch.zeros(batch_size, features.size(2), features.size(3), device=device)
            
            for i in range(batch_size):
                # Get weights for predicted class
                fc_weights = self.model.fc.weight[class_idx[i]]
                
                # Weighted combination
                for j in range(features.size(1)):
                    cam[i] += fc_weights[j] * features[i, j]
                
                # Apply ReLU and normalize
                cam[i] = F.relu(cam[i])
                if cam[i].max() > 0:
                    cam[i] = cam[i] / cam[i].max()
            
            return cam, output
    
    def _generate_grad_cam(self, input_tensor, class_idx=None):
        """Grad-CAM generation for better localization"""
        self.gradients = []
        self.activations = []
        
        input_tensor.requires_grad_()
        output, features = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass for gradients
        self.model.zero_grad()
        output[0, class_idx[0]].backward(retain_graph=True)
        
        if len(self.gradients) > 0 and len(self.activations) > 0:
            gradients = self.gradients[-1]
            activations = self.activations[-1]
            
            # Compute weights as global average of gradients
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            
            # Weighted combination
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Normalize
            cam = cam.squeeze(1)
            for i in range(cam.size(0)):
                if cam[i].max() > 0:
                    cam[i] = cam[i] / cam[i].max()
        else:
            # Fallback to standard CAM
            return self._generate_standard_cam(input_tensor, class_idx)
        
        return cam, output.detach()
    
    def visualize_comparison(self, input_tensor, save_path=None):
        """Compare standard CAM vs Grad-CAM"""
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Generate both types of CAMs
        cam_standard, output = self.generate_cam(input_tensor, method='standard')
        cam_grad, _ = self.generate_cam(input_tensor, method='grad_cam')
        
        batch_size = min(input_tensor.size(0), 4)  # Show max 4 samples
        fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4*batch_size))
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            # Original image
            img = input_tensor[i, 0].cpu().detach().numpy()
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Standard CAM
            cam_std = cam_standard[i].cpu().detach().numpy()
            axes[i, 1].imshow(cam_std, cmap='jet')
            axes[i, 1].set_title('Standard CAM')
            axes[i, 1].axis('off')
            
            # Grad-CAM
            cam_g = cam_grad[i].cpu().detach().numpy()
            axes[i, 2].imshow(cam_g, cmap='jet')
            axes[i, 2].set_title('Grad-CAM')
            axes[i, 2].axis('off')
            
            # Standard CAM Overlay
            axes[i, 3].imshow(img, cmap='gray', alpha=0.7)
            axes[i, 3].imshow(cam_std, cmap='jet', alpha=0.5)
            axes[i, 3].set_title('Standard Overlay')
            axes[i, 3].axis('off')
            
            # Grad-CAM Overlay
            axes[i, 4].imshow(img, cmap='gray', alpha=0.7)
            axes[i, 4].imshow(cam_g, cmap='jet', alpha=0.5)
            axes[i, 4].set_title('Grad-CAM Overlay')
            axes[i, 4].axis('off')
        
        plt.suptitle('CAM Comparison: Standard vs Grad-CAM', fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

# Training function with improved techniques
def train_improved_model():
    """Train the improved high-resolution CNN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Data loading with augmentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Model, loss, optimizer
    model = ImprovedHighResCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    # Training loop
    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        accuracy = 100. * correct / total
        print(f'Epoch {epoch}: Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    return model, test_loader

# Example usage
if __name__ == "__main__":
    print("Training improved high-resolution CNN...")
    model, test_loader = train_improved_model()
    
    print("Generating enhanced CAM visualizations...")
    visualizer = EnhancedCAMVisualizer(model)
    
    # Get test samples
    data_iter = iter(test_loader)
    test_data, test_targets = next(data_iter)
    sample_data = test_data[:4]
    
    # Compare CAM methods
    visualizer.visualize_comparison(sample_data)
    
    print("Enhanced CAM visualization complete!")
    print("Key improvements:")
    print("- Larger kernels (5x5, 7x7) for better spatial context")
    print("- Spatial attention mechanism")
    print("- Skip connections for gradient flow")
    print("- Grad-CAM comparison for better localization")