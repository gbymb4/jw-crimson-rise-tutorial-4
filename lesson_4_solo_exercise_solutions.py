# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 20:16:39 2025

@author: Gavin
"""

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader  
  
IMAGE_SIZE = 14  
  
# Channel Attention Module  
class ChannelAttentionModule(nn.Module):  
    def __init__(self, channels, reduction=16):  
        super(ChannelAttentionModule, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)  
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)  
        self.sigmoid = nn.Sigmoid()  
  
    def forward(self, x):  
        b, c, _, _ = x.size()  
        avg = self.avg_pool(x).view(b, c)  
        attention = self.fc2(F.relu(self.fc1(avg)))  
        attention = self.sigmoid(attention).view(b, c, 1, 1)  
        return x * attention.expand_as(x)  
  
# Spatial Attention Module  
class SpatialAttentionModule(nn.Module):  
    def __init__(self, kernel_size=7):  
        super(SpatialAttentionModule, self).__init__()  
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)  
        self.sigmoid = nn.Sigmoid()  
  
    def forward(self, x):  
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        combined = torch.cat([avg_out, max_out], dim=1)  
        attention = self.sigmoid(self.conv(combined))  
        return x * attention.expand_as(x)  
  
# CNN with attention mechanisms  
class AttentionCNN(nn.Module):  
    def __init__(self, num_classes=10):  
        super(AttentionCNN, self).__init__()  
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)  
  
        # Attention Modules  
        self.channel_attention = ChannelAttentionModule(64)  
        self.spatial_attention = SpatialAttentionModule()  
  
        self.fc = nn.Linear(64 * (IMAGE_SIZE ** 2), num_classes)  
  
    def forward(self, x):  
        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.relu(self.bn2(self.conv2(x)))  
  
        # Apply channel attention  
        x = self.channel_attention(x)  
  
        # Apply spatial attention  
        x = self.spatial_attention(x)  
  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
        return x  
  
# Training and evaluation  
def train_and_evaluate():  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
  
    # Data preparation with resolution reduction  
    transform = transforms.Compose([  
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Reduce resolution  
        transforms.ToTensor()  
    ])  
  
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)  
    test_dataset = datasets.MNIST('data', train=False, transform=transform)  
  
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  
  
    # Model, loss, optimizer  
    model = AttentionCNN().to(device)  
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
  
    # Training loop  
    model.train()  
    for epoch in range(10):  
        total_loss = 0  
        correct = 0  
        total = 0  
  
        for data, target in train_loader:  
            data, target = data.to(device), target.to(device)  
  
            optimizer.zero_grad()  
            output = model(data)  
            loss = criterion(output, target)  
            loss.backward()  
            optimizer.step()  
  
            total_loss += loss.item()  
            pred = output.argmax(dim=1)  
            correct += pred.eq(target).sum().item()  
            total += target.size(0)  
  
        accuracy = 100. * correct / total  
        print(f'Epoch {epoch}: Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')  
  
    # Evaluate on test set  
    model.eval()  
    correct = 0  
    total = 0  
    with torch.no_grad():  
        for data, target in test_loader:  
            data, target = data.to(device), target.to(device)  
            output = model(data)  
            pred = output.argmax(dim=1)  
            correct += pred.eq(target).sum().item()  
            total += target.size(0)  
  
    accuracy = 100. * correct / total  
    print(f'Test Accuracy: {accuracy:.2f}%')  
  
if __name__ == "__main__":  
    train_and_evaluate()  
