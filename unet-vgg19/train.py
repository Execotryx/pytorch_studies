import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import numpy.typing as npt
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Any
import logging
from unet_vgg19 import UNetVGG19


class SegmentationDataset(Dataset):
    """Custom dataset for segmentation tasks."""
    
    def __init__(self, images_dir: str, masks_dir: str, transform: transforms.Compose | None = None) -> None:
        self.images_dir: Path = Path(images_dir)
        self.masks_dir: Path = Path(masks_dir)
        self.transform: transforms.Compose | None = transform
        
        # Get all image files
        self.image_files: list[str] = sorted([f for f in os.listdir(images_dir) 
                                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        
        image_path: Path = self.images_dir / img_name
        mask_path: Path = self.masks_dir / mask_name
        
        # Load images using OpenCV
        image_bgr: Any = cv2.imread(str(image_path))
        image_np: Any = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        mask_np: Any = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        
        image: torch.Tensor
        mask: torch.Tensor
        
        if self.transform:
            # Apply transform to image
            image = self.transform(image_np)
            # Convert mask to tensor
            mask = torch.from_numpy(mask_np).float().unsqueeze(0) / 255.0
        else:
            # Convert to tensors
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask_np).float().unsqueeze(0) / 255.0
        
        # Ensure mask values are 0 or 1
        mask = (mask > 0.5).float()
        
        return image, mask


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth: float = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        total = predictions.sum() + targets.sum()
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        
        # Return dice loss
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combination of BCE and Dice loss."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight: float = bce_weight
        self.dice_weight: float = dice_weight
        self.bce_loss: nn.BCELoss = nn.BCELoss()
        self.dice_loss: DiceLoss = DiceLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def calculate_iou(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Intersection over Union (IoU) metric."""
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    intersection = (predictions * targets).sum().item()
    union = (predictions + targets).clamp(0, 1).sum().item()
    
    if union == 0:
        return 1.0  # Perfect score if both are empty
    
    return intersection / union


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss: float = 0.0
    total_iou: float = 0.0
    num_batches: int = len(dataloader)
    
    progress_bar: tqdm = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        total_loss += loss.item()
        iou = calculate_iou(outputs, masks)
        total_iou += iou
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{iou:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_iou


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> tuple[float, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss: float = 0.0
    total_iou: float = 0.0
    num_batches: int = len(dataloader)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            total_loss += loss.item()
            iou = calculate_iou(outputs, masks)
            total_iou += iou
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_iou


def plot_training_history(train_losses: list[float], val_losses: list[float], 
                         train_ious: list[float], val_ious: list[float]) -> None:
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot IoU
    ax2.plot(train_ious, label='Train IoU')
    ax2.plot(val_ious, label='Validation IoU')
    ax2.set_title('Training and Validation IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main() -> None:
    # Configuration
    config: dict[str, Any] = {
        'data_dir': 'path/to/your/dataset',  # Update this path
        'images_dir': 'path/to/your/dataset/images',  # Update this path
        'masks_dir': 'path/to/your/dataset/masks',    # Update this path
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'num_classes': 1,  # Binary segmentation
        'image_size': (256, 256),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'early_stopping_patience': 10
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger: logging.Logger = logging.getLogger(__name__)
    
    logger.info(f"Using device: {config['device']}")
    logger.info(f"Training configuration: {config}")
    
    # Data transforms
    train_transform: transforms.Compose = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform: transforms.Compose = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets (you'll need to split your data into train/val)
    train_dataset: SegmentationDataset = SegmentationDataset(
        images_dir=os.path.join(config['images_dir'], 'train'),
        masks_dir=os.path.join(config['masks_dir'], 'train'),
        transform=train_transform
    )
    
    val_dataset: SegmentationDataset = SegmentationDataset(
        images_dir=os.path.join(config['images_dir'], 'val'),
        masks_dir=os.path.join(config['masks_dir'], 'val'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model: UNetVGG19 = UNetVGG19(num_classes=config['num_classes'], pretrained=True)
    model = model.to(config['device'])
    
    # Loss function and optimizer
    criterion: CombinedLoss = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_ious: list[float] = []
    val_ious: list[float] = []
    best_val_iou: float = 0.0
    patience_counter: int = 0
    epoch: int = 0
    val_iou: float = 0.0
    
    logger.info("Starting training...")
    
    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss: float
        train_iou: float
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        
        # Validate
        val_loss: float
        val_iou: float
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, config['device'])
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            logger.info(f"New best model saved with IoU: {best_val_iou:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("Training completed!")
    logger.info(f"Best validation IoU: {best_val_iou:.4f}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_ious, val_ious)
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_iou': val_iou,
        'config': config
    }, os.path.join(config['save_dir'], 'final_model.pth'))


if __name__ == "__main__":
    main()