import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast_mode
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import numpy.typing as npt
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Any
import logging
import math
from unet_vgg19 import UNetVGG19


def get_cosine_schedule_with_warmup(optimizer: optim.Optimizer, num_warmup_steps: int, 
                                    num_training_steps: int, num_cycles: float = 0.5,
                                    last_epoch: int = -1) -> optim.lr_scheduler.LambdaLR:
    """Create a schedule with a learning rate that decreases following cosine after a warmup period."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


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
        
        if self.transform:
            # Apply transform to image
            image = self.transform(image_np)
            
            # Get target size from transformed image
            target_size = (image.shape[1], image.shape[2])  # H, W
            
            # Resize mask to match transformed image size
            mask_resized = cv2.resize(mask_np, (target_size[1], target_size[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Convert mask to tensor with proper memory layout
            mask = torch.from_numpy(mask_resized.copy()).float().unsqueeze(0) / 255.0
        else:
            # Convert to tensors
            image = torch.from_numpy(image_np.copy()).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask_np.copy()).float().unsqueeze(0) / 255.0
        
        # Ensure mask values are 0 or 1
        mask = (mask > 0.5).float()
        
        # Ensure both tensors are contiguous
        image = image.contiguous()
        mask = mask.contiguous()
        
        return image, mask


class OxfordPetsDataset(Dataset):
    """Wrapper for Oxford-IIIT Pet Dataset with binary segmentation masks."""
    
    def __init__(self, root: str, split: str = 'trainval', transform: transforms.Compose | None = None, download: bool = True) -> None:
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types='segmentation',
            download=download
        )
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[idx]
        
        # Convert trimap to binary mask (1=foreground, 2=boundary, 3=background)
        # We'll treat foreground (1) and boundary (2) as the pet (1), background (3) as 0
        mask_np = np.array(mask)
        binary_mask = (mask_np < 3).astype(np.float32)
        
        if self.transform:
            # Apply transform to image
            image_tensor = self.transform(image)
            
            # Get target size from transformed image
            target_size = (image_tensor.shape[1], image_tensor.shape[2])  # H, W
            
            # Resize mask to match transformed image size
            mask_resized = cv2.resize(binary_mask, (target_size[1], target_size[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Convert mask to tensor with proper memory layout
            mask_tensor = torch.from_numpy(mask_resized.copy()).float().unsqueeze(0).contiguous()
        else:
            # Convert PIL image to numpy
            image_np = np.array(image)
            # Convert to tensors
            image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(binary_mask.copy()).float().unsqueeze(0)
        
        # Ensure both tensors are contiguous
        image_tensor = image_tensor.contiguous()
        mask_tensor = mask_tensor.contiguous()
        
        return image_tensor, mask_tensor


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth: float = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten tensors (use reshape instead of view for better memory efficiency)
        predictions_flat = predictions.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Calculate intersection and union
        intersection = (predictions_flat * targets_flat).sum()
        total = predictions_flat.sum() + targets_flat.sum()
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        
        # Return dice loss
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combination of BCE and Dice loss (AMP-safe)."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight: float = bce_weight
        self.dice_weight: float = dice_weight
        # Use BCEWithLogitsLoss for AMP safety (expects logits, not probabilities)
        self.bce_loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.dice_loss: DiceLoss = DiceLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCEWithLogitsLoss expects logits
        bce = self.bce_loss(predictions, targets)
        # Dice loss needs probabilities, so apply sigmoid
        predictions_probs = torch.sigmoid(predictions)
        dice = self.dice_loss(predictions_probs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def calculate_iou(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Intersection over Union (IoU) metric."""
    # Apply sigmoid to convert logits to probabilities
    predictions_probs = torch.sigmoid(predictions)
    predictions_binary = (predictions_probs > threshold).float()
    targets_binary = (targets > threshold).float()
    
    intersection = (predictions_binary * targets_binary).sum().item()
    union = (predictions_binary + targets_binary).clamp(0, 1).sum().item()
    
    if union == 0:
        return 1.0  # Perfect score if both are empty
    
    return intersection / union


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, 
                scaler: GradScaler | None = None,
                gradient_accumulation_steps: int = 1,
                scheduler: Any | None = None,
                scheduler_step_on: str = 'epoch') -> tuple[float, float]:
    """Train for one epoch with optional mixed precision and gradient accumulation."""
    model.train()
    total_loss: float = 0.0
    total_iou: float = 0.0
    num_batches: int = len(dataloader)
    use_amp = scaler is not None
    
    progress_bar: tqdm = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Forward pass with optional mixed precision
        with autocast_mode.autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights after accumulation steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            # Step scheduler if configured for step-level updates
            if scheduler is not None and scheduler_step_on == 'step':
                scheduler.step()
        
        # Calculate metrics (detach to save memory)
        with torch.no_grad():
            total_loss += loss.item() * gradient_accumulation_steps
            iou = calculate_iou(outputs.detach(), masks)
            total_iou += iou
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'IoU': f'{iou:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_iou


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                   device: torch.device, use_amp: bool = False) -> tuple[float, float]:
    """Validate for one epoch with optional mixed precision."""
    model.eval()
    total_loss: float = 0.0
    total_iou: float = 0.0
    num_batches: int = len(dataloader)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for images, masks in progress_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            with autocast_mode.autocast(device_type='cuda', enabled=use_amp):
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


def print_training_history(train_losses: list[float], val_losses: list[float], 
                          train_ious: list[float], val_ious: list[float]) -> None:
    """Print training history as ASCII table."""
    print("\n" + "="*70)
    print("TRAINING HISTORY".center(70))
    print("="*70)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Train IoU':>12} | {'Val IoU':>12}")
    print("-"*70)
    
    for i in range(len(train_losses)):
        print(f"{i+1:6d} | {train_losses[i]:12.6f} | {val_losses[i]:12.6f} | {train_ious[i]:12.6f} | {val_ious[i]:12.6f}")
    
    print("="*70)
    print(f"Best Train Loss: {min(train_losses):.6f} at epoch {train_losses.index(min(train_losses)) + 1}")
    print(f"Best Val Loss:   {min(val_losses):.6f} at epoch {val_losses.index(min(val_losses)) + 1}")
    print(f"Best Train IoU:  {max(train_ious):.6f} at epoch {train_ious.index(max(train_ious)) + 1}")
    print(f"Best Val IoU:    {max(val_ious):.6f} at epoch {val_ious.index(max(val_ious)) + 1}")
    print("="*70 + "\n")


def main() -> None:
    # Configuration
    config: dict[str, Any] = {
        # Dataset settings
        'dataset': 'oxford_pets',  # 'oxford_pets' or 'custom'
        'data_dir': './data',  # Oxford Pets will be downloaded here
        'train_val_split': 0.8,  # 80% train, 20% validation for Oxford Pets
        
        # Custom dataset paths (only used if dataset='custom')
        'images_dir': 'path/to/your/dataset/images',
        'masks_dir': 'path/to/your/dataset/masks',
        
        # Training settings
        'batch_size': 2,  # Reduced for higher resolution (384x384)
        'learning_rate': 3e-4,  # Higher initial LR (will be warmed up)
        'min_lr': 1e-6,  # Minimum learning rate for cosine annealing
        'num_epochs': 100,  # Increased for better convergence with cosine schedule
        'num_classes': 1,  # Binary segmentation
        'image_size': (384, 384),  # Increased from 256x256 for better accuracy
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': 'checkpoints',
        'early_stopping_patience': 25,  # Increased patience for cosine schedule
        
        # Memory optimization settings
        'use_amp': True,  # Automatic Mixed Precision (reduces memory by ~40%)
        'gradient_accumulation_steps': 8,  # Effective batch size: 2 * 8 = 16
        'num_workers': 2,  # Reduced for memory efficiency
        'pin_memory': True,  # Keep True if using CUDA
        
        # Learning rate schedule settings
        'use_cosine_schedule': True,  # Use cosine annealing with warmup
        'warmup_epochs': 5,  # Warmup for first 5 epochs
        'use_onecycle': False,  # Alternative: OneCycleLR (set to True to use instead)
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
    
    # Create datasets
    if config['dataset'] == 'oxford_pets':
        logger.info("Using Oxford-IIIT Pet Dataset (will download if needed)...")
        
        # Load full dataset
        full_dataset = OxfordPetsDataset(
            root=config['data_dir'],
            split='trainval',
            transform=train_transform,
            download=True
        )
        
        # Split into train and validation
        train_size = int(config['train_val_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset_temp = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create validation dataset with different transform
        val_full_dataset = OxfordPetsDataset(
            root=config['data_dir'],
            split='trainval',
            transform=val_transform,
            download=False
        )
        
        # Use same indices for validation
        val_dataset = torch.utils.data.Subset(val_full_dataset, val_dataset_temp.indices)
    else:
        logger.info("Using custom dataset...")
        train_dataset = SegmentationDataset(
            images_dir=os.path.join(config['images_dir'], 'train'),
            masks_dir=os.path.join(config['masks_dir'], 'train'),
            transform=train_transform
        )
        
        val_dataset = SegmentationDataset(
            images_dir=os.path.join(config['images_dir'], 'val'),
            masks_dir=os.path.join(config['masks_dir'], 'val'),
            transform=val_transform
        )
    
    # Create data loaders with memory-efficient settings
    device: torch.device = config['device']
    is_cuda = device.type == 'cuda'
    
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'] and is_cuda,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'] and is_cuda,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model: UNetVGG19 = UNetVGG19(num_classes=config['num_classes'], pretrained=True)
    model = model.to(config['device'])
    
    # Loss function and optimizer
    criterion: CombinedLoss = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    # Calculate total training steps
    steps_per_epoch = len(train_loader) // config['gradient_accumulation_steps']
    total_training_steps = steps_per_epoch * config['num_epochs']
    warmup_steps = steps_per_epoch * config['warmup_epochs']
    
    # Learning rate scheduler
    scheduler: Any
    scheduler_step_on: str
    
    if config['use_onecycle']:
        logger.info("Using OneCycleLR scheduler")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            total_steps=total_training_steps,
            pct_start=config['warmup_epochs'] / config['num_epochs'],
            anneal_strategy='cos',
            div_factor=25.0,  # initial_lr = max_lr/25
            final_div_factor=1e4  # min_lr = initial_lr/1e4
        )
        scheduler_step_on = 'step'  # Update every step
    elif config['use_cosine_schedule']:
        logger.info("Using Cosine schedule with warmup")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
        scheduler_step_on = 'step'  # Update every step
    else:
        logger.info("Using ReduceLROnPlateau scheduler")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        scheduler_step_on = 'epoch'  # Update every epoch
    
    # Initialize mixed precision scaler
    scaler: GradScaler | None = GradScaler() if config['use_amp'] and is_cuda else None
    
    # Log optimization settings
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION SETTINGS".center(60))
    logger.info("="*60)
    
    if config['use_amp'] and is_cuda:
        logger.info("✓ Automatic Mixed Precision (AMP) enabled")
    if config['gradient_accumulation_steps'] > 1:
        logger.info(f"✓ Gradient accumulation: {config['gradient_accumulation_steps']} steps")
        logger.info(f"✓ Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    else:
        logger.info(f"✓ Batch size: {config['batch_size']}")
    
    logger.info(f"✓ Initial learning rate: {config['learning_rate']:.2e}")
    logger.info(f"✓ Total training steps: {total_training_steps:,}")
    logger.info(f"✓ Warmup steps: {warmup_steps:,} ({config['warmup_epochs']} epochs)")
    logger.info(f"✓ Scheduler updates: {scheduler_step_on}-level")
    logger.info("="*60 + "\n")
    
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
    
    try:
        for epoch in range(config['num_epochs']):
            logger.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            
            # Train
            train_loss: float
            train_iou: float
            train_loss, train_iou = train_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler=scaler,
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                scheduler=scheduler,
                scheduler_step_on=scheduler_step_on
            )
            
            # Validate
            val_loss: float
            val_iou: float
            val_loss, val_iou = validate_epoch(
                model, val_loader, criterion, device,
                use_amp=config['use_amp'] and is_cuda
            )
            
            # Clear cache to free memory
            if is_cuda:
                torch.cuda.empty_cache()
            
            # Update learning rate (only for epoch-based schedulers)
            if scheduler_step_on == 'epoch':
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
        
        # Print training history
        print_training_history(train_losses, val_losses, train_ious, val_ious)
        
        # Save final model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_val_iou': val_iou,
            'config': config
        }, os.path.join(config['save_dir'], 'final_model.pth'))
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (Ctrl+C)")
        logger.info(f"Saving checkpoint at epoch {epoch+1}...")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_ious': train_ious,
                'val_ious': val_ious,
                'config': config
            }, os.path.join(config['save_dir'], 'interrupted_checkpoint.pth'))
            logger.info("Checkpoint saved successfully")
        except Exception as save_error:
            logger.error(f"Failed to save checkpoint: {save_error}")
    
    except RuntimeError as e:
        logger.error(f"\nRuntime error during training: {e}")
        if "out of memory" in str(e).lower():
            logger.error("Out of memory error detected!")
            logger.info("Suggestions:")
            logger.info("  - Reduce batch_size in config")
            logger.info("  - Increase gradient_accumulation_steps")
            logger.info("  - Reduce image_size")
            logger.info("  - Ensure use_amp is enabled")
        raise
    
    except Exception as e:
        logger.error(f"\nUnexpected error during training: {type(e).__name__}: {e}")
        logger.info("Attempting to save emergency checkpoint...")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'error': str(e),
                'config': config
            }, os.path.join(config['save_dir'], 'emergency_checkpoint.pth'))
            logger.info("Emergency checkpoint saved")
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")
        raise
    
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        
        # Clear CUDA cache
        if is_cuda:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        
        # Delete large objects to free memory
        try:
            del model
            del optimizer
            del train_loader
            del val_loader
            if scaler is not None:
                del scaler
            logger.info("Training resources released")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")
        
        # Final cache clear
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Cleanup completed")


if __name__ == "__main__":
    main()