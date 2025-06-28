# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 23:04:26 2025

@author: Gavin

HOMEWORK: Image Segmentation with PyTorch using PASCAL VOC Dataset (64x64)

BACKGROUND:
You've worked with CNNs for image classification, where the output is a single class label.
Image segmentation is different - instead of classifying the entire image, you need to 
classify EVERY PIXEL in the image. This means your output should be the same spatial 
dimensions as your input image, but with class predictions for each pixel.

TASK:
1. Design and implement a segmentation model architecture using only basic CNN and residual blocks
2. The training loop is provided for you - focus on understanding how it works
3. Your model should take images of size (3, 64, 64) and output (21, 64, 64)
4. Work with the real PASCAL VOC 2012 dataset (downscaled to 64x64 for speed)

DATASET:
PASCAL VOC 2012 - A classic segmentation dataset with 20 object classes + background
Classes include: person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat,
bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, tv/monitor
Images are downscaled to 64x64 to make training faster and even more challenging!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================================
# DATASET SETUP WITH VOC (64x64)
# ================================

def get_transforms(target_size=64, augment=True):
    """Get image and target transforms for VOC dataset"""
    
    if augment:
        # Training transforms with augmentation
        img_transform = transforms.Compose([
            transforms.Resize((target_size + 8, target_size + 8)),
            transforms.RandomCrop((target_size, target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])
        
        target_transform = transforms.Compose([
            transforms.Resize((target_size + 8, target_size + 8), 
                            interpolation=Image.NEAREST),
            transforms.RandomCrop((target_size, target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.squeeze(0).long()),  # Remove channel dim and convert to long
            transforms.Lambda(lambda x: torch.where(x == 255, 0, x))  # Convert ignore class to background
        ])
    else:
        # Validation transforms without augmentation
        img_transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor()
        ])
        
        target_transform = transforms.Compose([
            transforms.Resize((target_size, target_size), 
                            interpolation=Image.NEAREST),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.squeeze(0).long()),  # Remove channel dim and convert to long
            transforms.Lambda(lambda x: torch.where(x == 255, 0, x))  # Convert ignore class to background
        ])
    
    return img_transform, target_transform

def create_subset_indices(dataset_size, fraction, seed=42):
    """
    Create indices for a subset of the dataset
    
    Args:
        dataset_size (int): Total size of the dataset
        fraction (float): Fraction of dataset to use (0.0 to 1.0)
        seed (int): Random seed for reproducibility
    
    Returns:
        list: Indices for the subset
    """
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("Fraction must be between 0.0 and 1.0")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Calculate subset size
    subset_size = int(dataset_size * fraction)
    
    # Generate random indices
    all_indices = np.arange(dataset_size)
    subset_indices = np.random.choice(all_indices, size=subset_size, replace=False)
    
    return sorted(subset_indices.tolist())

def get_voc_datasets(root='./data', target_size=64, train_fraction=1.0, val_fraction=1.0, seed=42):
    """
    Download and prepare VOC datasets using built-in torchvision dataset
    
    Args:
        root (str): Root directory for dataset
        target_size (int): Target image size (default: 64)
        train_fraction (float): Fraction of training data to use (0.0 to 1.0)
        val_fraction (float): Fraction of validation data to use (0.0 to 1.0)
        seed (int): Random seed for subset selection
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    print("Preparing VOC datasets...")
    print(f"Target image size: {target_size}x{target_size}")
    print(f"Training fraction: {train_fraction:.2f}")
    print(f"Validation fraction: {val_fraction:.2f}")
    if train_fraction < 1.0 or val_fraction < 1.0:
        print("This will download ~2GB if not already present")
    
    # Get transforms
    train_img_transform, train_target_transform = get_transforms(target_size, augment=True)
    val_img_transform, val_target_transform = get_transforms(target_size, augment=False)
    
    # Create full datasets using built-in VOCSegmentation
    full_train_dataset = datasets.VOCSegmentation(
        root=root,
        year='2012',
        image_set='train',
        download=True,
        transform=train_img_transform,
        target_transform=train_target_transform
    )
    
    full_val_dataset = datasets.VOCSegmentation(
        root=root,
        year='2012', 
        image_set='val',
        download=True,
        transform=val_img_transform,
        target_transform=val_target_transform
    )
    
    # Create subsets if fractions are less than 1.0
    if train_fraction < 1.0:
        train_indices = create_subset_indices(len(full_train_dataset), train_fraction, seed)
        train_dataset = Subset(full_train_dataset, train_indices)
        print(f"Using {len(train_indices)}/{len(full_train_dataset)} training samples")
    else:
        train_dataset = full_train_dataset
        print(f"Using all {len(full_train_dataset)} training samples")
    
    if val_fraction < 1.0:
        val_indices = create_subset_indices(len(full_val_dataset), val_fraction, seed)
        val_dataset = Subset(full_val_dataset, val_indices)
        print(f"Using {len(val_indices)}/{len(full_val_dataset)} validation samples")
    else:
        val_dataset = full_val_dataset
        print(f"Using all {len(full_val_dataset)} validation samples")
    
    print(f"Final train samples: {len(train_dataset)}")
    print(f"Final val samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

# VOC class names for reference
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# ================================
# YOUR TASK: IMPLEMENT THE MODEL
# ================================

class ResidualBlock(nn.Module):
    """
    Basic residual block for segmentation
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out

class SegmentationModel(nn.Module):
    """
    Segmentation model using residual blocks and basic CNN architecture
    
    STRATEGY for 64x64:
    1. Use very minimal downsampling (at most 2x) to preserve spatial information
    2. Use residual blocks to go deeper without losing gradients
    3. Use dilated convolutions to increase receptive field without downsampling
    4. Simple upsampling at the end to restore resolution
    
    ARCHITECTURE:
    - Initial conv: 64x64 -> 64x64
    - Residual blocks with same spatial size: 64x64
    - Optional single downsample: 64x64 -> 32x32
    - More residual blocks at 32x32
    - Upsample back to 64x64
    - Final classification layer
    """
    
    def __init__(self, num_classes=21):
        super(SegmentationModel, self).__init__()
        self.num_classes = num_classes
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # TODO: Implement your architecture here
        
        
        self.out_conv = nn.LazyConv2d(self.num_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass for segmentation
        
        Input: (batch_size, 3, 64, 64)
        Output: (batch_size, 21, 64, 64)
        """
        
        # Initial feature extraction
        x = self.initial_conv(x)  # (B, 64, 64, 64)
        
        # TODO: Execute your layers
        
        # Output layer
        x = F.sigmoid(self.out_conv(x))
        
        return x

# ================================
# TRAINING UTILITIES
# ================================

def calculate_iou(pred_mask, true_mask, num_classes, ignore_index=0):
    """
    Calculate Intersection over Union (IoU) metric
    
    For VOC, we typically ignore the background class in mIoU calculation
    """
    ious = []
    pred_mask = pred_mask.cpu().numpy()
    true_mask = true_mask.cpu().numpy()
    
    for cls in range(1 if ignore_index == 0 else 0, num_classes):  # Skip background
        if cls == ignore_index:
            continue
            
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)
        
        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()
        
        if union == 0:
            continue  # Skip classes not present in this batch
        else:
            iou = intersection / union
            ious.append(iou)
    
    return np.mean(ious) if ious else 0.0

def calculate_pixel_accuracy(pred_mask, true_mask):
    """Calculate pixel-wise accuracy"""
    correct = (pred_mask == true_mask).sum().item()
    total = true_mask.numel()
    return correct / total

def visualize_predictions(model, dataset, device, num_samples=4):
    """Visualize model predictions on VOC data"""
    model.eval()
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 10))
    
    # VOC colormap for visualization
    def get_voc_colormap():
        colormap = np.zeros((256, 3), dtype=np.uint8)
        for i in range(21):
            colormap[i] = np.array([
                (i * 12) % 256,
                (i * 34 + 100) % 256, 
                (i * 56 + 200) % 256
            ])
        return colormap
    
    colormap = get_voc_colormap()
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            image, true_mask = dataset[i]
            image_batch = image.unsqueeze(0).to(device)
            
            # Get prediction
            pred_logits = model(image_batch)
            pred_mask = torch.argmax(pred_logits, dim=1).squeeze().cpu()
            
            # Denormalize image for display
            img_display = image.clone()
            img_display = img_display * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = img_display + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img_display = torch.clamp(img_display, 0, 1)
            img_display = img_display.permute(1, 2, 0).numpy()
            
            # Apply colormap to masks
            true_mask_colored = colormap[true_mask.numpy()]
            pred_mask_colored = colormap[pred_mask.numpy()]
            
            # Calculate metrics
            iou = calculate_iou(pred_mask.unsqueeze(0), true_mask.unsqueeze(0), 21)
            pixel_acc = calculate_pixel_accuracy(pred_mask, true_mask)
            
            # Plot
            axes[0, i].imshow(img_display)
            axes[0, i].set_title('Original Image')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(true_mask_colored)
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(pred_mask_colored)
            axes[2, i].set_title(f'Prediction\nIoU: {iou:.3f}, Acc: {pixel_acc:.3f}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_dataset_samples(dataset, num_samples=6):
    """Show some dataset samples to understand the data"""
    fig, axes = plt.subplots(2, num_samples, figsize=(18, 6))
    
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(21):
        colormap[i] = np.array([
            (i * 12) % 256,
            (i * 34 + 100) % 256, 
            (i * 56 + 200) % 256
        ])
    
    for i in range(num_samples):
        image, mask = dataset[i]
        
        # Denormalize image
        img_display = image.clone()
        img_display = img_display * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = img_display + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img_display = torch.clamp(img_display, 0, 1)
        img_display = img_display.permute(1, 2, 0).numpy()
        
        mask_colored = colormap[mask.numpy()]
        
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask_colored)
        unique_classes = torch.unique(mask)
        class_names = [VOC_CLASSES[c] for c in unique_classes[:5]]  # Show first 5
        axes[1, i].set_title(f'Mask\n{", ".join(class_names)}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# ================================
# MAIN TRAINING LOOP
# ================================

def main(train_fraction=1.0, val_fraction=1.0, dataset_seed=42):
    """
    Main training function with dataset fraction control
    
    Args:
        train_fraction (float): Fraction of training data to use (0.0 to 1.0)
        val_fraction (float): Fraction of validation data to use (0.0 to 1.0)
        dataset_seed (int): Random seed for dataset subset selection
    """
    # Hyperparameters
    batch_size = 32  # Can use larger batch size with 64x64
    learning_rate = 0.001
    num_epochs = 30
    num_classes = 21
    target_size = 64
    
    # Create datasets with fraction control
    print("Setting up PASCAL VOC datasets...")
    train_dataset, val_dataset = get_voc_datasets(
        target_size=target_size, 
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=dataset_seed
    )
    
    # Show some samples to understand the data
    print("\nDataset samples:")
    show_dataset_samples(train_dataset, num_samples=6)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = SegmentationModel(num_classes=num_classes).to(device)
    
    # Loss function and optimizer
    # Use class weights to handle class imbalance (background is very common)
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore undefined pixels
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    val_ious = []
    val_pixel_accs = []
    
    print(f"\nStarting training on {target_size}x{target_size} images...")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)}")
    print(f"Dataset fractions - Train: {train_fraction:.2f}, Val: {val_fraction:.2f}")
    
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        total_pixel_acc = 0.0
        num_val_samples = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                pred_masks = torch.argmax(outputs, dim=1)
                for i in range(images.size(0)):
                    iou = calculate_iou(pred_masks[i:i+1], masks[i:i+1], num_classes)
                    pixel_acc = calculate_pixel_accuracy(pred_masks[i], masks[i])
                    total_iou += iou
                    total_pixel_acc += pixel_acc
                    num_val_samples += 1
        
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = total_iou / num_val_samples
        avg_pixel_acc = total_pixel_acc / num_val_samples
        
        val_losses.append(avg_val_loss)
        val_ious.append(avg_iou)
        val_pixel_accs.append(avg_pixel_acc)
        
        # Update learning rate
        scheduler.step(avg_iou)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val mIoU: {avg_iou:.4f}')
        print(f'  Val Pixel Acc: {avg_pixel_acc:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if avg_iou > best_iou:
            best_iou = avg_iou
            model_name = f'best_segmentation_model_64x64_train{train_fraction:.2f}_val{val_fraction:.2f}.pth'
            torch.save(model.state_dict(), model_name)
            print(f'  New best mIoU! Model saved as {model_name}')
        
        print('-' * 60)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss\n(Train: {train_fraction:.2f}, Val: {val_fraction:.2f})')
    
    plt.subplot(1, 3, 2)
    plt.plot(val_ious)
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Validation mIoU')
    
    plt.subplot(1, 3, 3)
    plt.plot(val_pixel_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.title('Validation Pixel Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    # Show final predictions
    print("\nFinal Predictions:")
    visualize_predictions(model, val_dataset, device, num_samples=4)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print(f"Dataset fractions - Train: {train_fraction:.2f}, Val: {val_fraction:.2f}")
    print(f"Best Validation mIoU: {best_iou:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Validation mIoU: {val_ious[-1]:.4f}")
    print(f"Final Pixel Accuracy: {val_pixel_accs[-1]:.4f}")
    print("="*60)
    
    return model

if __name__ == "__main__":
    print("="*60)
    print("PASCAL VOC IMAGE SEGMENTATION HOMEWORK - 64x64 VERSION")
    print("="*60)
    print("\nCHALLENGE: 64x64 Resolution with Residual CNN")
    print("This is extremely difficult! Real VOC images are ~500x500.")
    print("Your model uses only basic CNN and residual blocks (no U-Net).")
    print("\nArchitecture Features:")
    print("✓ Residual blocks for deeper networks")
    print("✓ Minimal downsampling to preserve spatial info")
    print("✓ Dilated convolutions for larger receptive field")
    print("✓ Simple upsampling approach")
    print("\nDataset Fraction Control:")
    print("✓ Control training and validation dataset sizes")
    print("✓ Useful for quick experimentation and debugging")
    print("✓ Reproducible subset selection with random seed")
    print("\nInstructions:")
    print("1. The SegmentationModel class is implemented with residual blocks")
    print("2. You can modify the architecture by commenting/uncommenting sections")
    print("3. Two approaches provided: with/without downsampling")
    print("4. Run this script to train on real PASCAL VOC data")
    print("5. Adjust train_fraction and val_fraction for smaller datasets")
    print("\nSuccess Criteria for 64x64:")
    print("- Model should train without errors")
    print("- Validation mIoU should improve over epochs")
    print("- Target: mIoU > 0.15 (very challenging at 64x64!)")
    print("- Good solution: mIoU > 0.25")
    print("- Excellent solution: mIoU > 0.35")
    print("="*60)
    
    # Configuration: Adjust these parameters as needed
    TRAIN_FRACTION = 0.1  # Use 10% of training data (change to 1.0 for full dataset)
    VAL_FRACTION = 0.1    # Use 10% of validation data (change to 1.0 for full dataset)
    SEED = 42             # Random seed for reproducible subset selection
    
    print(f"\nDataset Configuration:")
    print(f"Training fraction: {TRAIN_FRACTION:.1%}")
    print(f"Validation fraction: {VAL_FRACTION:.1%}")
    print(f"Random seed: {SEED}")
    
    # Test dataset loading first
    try:
        print("\nTesting dataset loading...")
        train_dataset, val_dataset = get_voc_datasets(
            target_size=64, 
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            seed=SEED
        )
        print("✓ Dataset loaded successfully!")
        print(f"Sample image shape: {train_dataset[0][0].shape}")
        print(f"Sample mask shape: {train_dataset[0][1].shape}")
        print(f"Classes in first mask: {torch.unique(train_dataset[0][1])}")
        
        # Start training
        model = main(
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            dataset_seed=SEED
        )
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have internet connection for initial download.")