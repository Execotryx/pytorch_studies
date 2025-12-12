import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import transforms
import cv2
import numpy as np
import numpy.typing as npt
import argparse
import os
import sys
from pathlib import Path
from typing import Any
from tqdm import tqdm
from unet_vgg19 import UNetVGG19


def load_model(checkpoint_path: str, device: torch.device, use_compile: bool = True) -> Any:
    """Load trained model from checkpoint with optimizations."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    
    # Get model configuration from checkpoint
    config = checkpoint.get('config', {'num_classes': 1})
    
    # Create model
    model = UNetVGG19(num_classes=config['num_classes'], pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Apply torch.compile for faster inference (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile for faster inference")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}. Using eager mode.")
    
    # Optimize for inference
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Best validation IoU: {checkpoint.get('best_val_iou', 'N/A')}")
    
    return model


# Cache normalization tensors to avoid recreation
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def preprocess_image(image_path: str, image_size: tuple[int, int] = (256, 256)) -> tuple[torch.Tensor, Any]:
    """Preprocess input image for inference with optimizations."""
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()
    
    # Resize image
    image_resized = cv2.resize(image_rgb, image_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor and normalize (optimized)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().div_(255.0)
    
    # Apply ImageNet normalization using cached tensors
    image_tensor = (image_tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_image

def preprocess_batch(image_paths: list[str], image_size: tuple[int, int] = (256, 256)) -> tuple[torch.Tensor, list[Any]]:
    """Preprocess multiple images into a batch for efficient inference."""
    batch_tensors = []
    original_images = []
    
    for image_path in image_paths:
        try:
            tensor, original = preprocess_image(image_path, image_size)
            batch_tensors.append(tensor)
            original_images.append(original)
        except Exception as e:
            print(f"Warning: Skipping {image_path} due to error: {e}")
    
    if not batch_tensors:
        raise ValueError("No valid images to process")
    
    # Stack into single batch tensor
    batch_tensor = torch.cat(batch_tensors, dim=0)
    
    return batch_tensor, original_images


def postprocess_mask(mask_tensor: torch.Tensor, threshold: float = 0.5) -> npt.NDArray[np.uint8]:
    """Postprocess model output to binary mask with optimizations."""
    # Remove batch dimension and convert to numpy (single operation)
    mask = mask_tensor.squeeze().cpu().numpy()
    
    # Apply threshold (optimized)
    binary_mask = (mask > threshold).astype(np.uint8)
    binary_mask *= 255
    
    return binary_mask


def print_inference_results(predicted_mask: npt.NDArray[np.uint8], 
                           confidence_map: npt.NDArray[np.float32], 
                           image_name: str = "Image") -> None:
    """Print inference results in console-friendly format."""
    # Calculate statistics
    total_pixels = predicted_mask.size
    foreground_pixels = np.sum(predicted_mask > 0)
    background_pixels = total_pixels - foreground_pixels
    foreground_ratio = (foreground_pixels / total_pixels) * 100
    
    # Confidence statistics
    mean_confidence = np.mean(confidence_map)
    max_confidence = np.max(confidence_map)
    min_confidence = np.min(confidence_map)
    std_confidence = np.std(confidence_map)
    
    # Foreground confidence (where mask is 1)
    fg_mask = predicted_mask > 0
    if np.any(fg_mask):
        fg_confidence = np.mean(confidence_map[fg_mask])
    else:
        fg_confidence = 0.0
    
    # Print results
    print("\n" + "="*60)
    print(f"INFERENCE RESULTS: {image_name}".center(60))
    print("="*60)
    
    print(f"\n{'Segmentation Statistics':^60}")
    print("-"*60)
    print(f"  Total Pixels:       {total_pixels:>12,}")
    print(f"  Foreground Pixels:  {foreground_pixels:>12,}  ({foreground_ratio:>5.2f}%)")
    print(f"  Background Pixels:  {background_pixels:>12,}  ({100-foreground_ratio:>5.2f}%)")
    
    print(f"\n{'Confidence Statistics':^60}")
    print("-"*60)
    print(f"  Mean Confidence:    {mean_confidence:>12.4f}")
    print(f"  Std Confidence:     {std_confidence:>12.4f}")
    print(f"  Min Confidence:     {min_confidence:>12.4f}")
    print(f"  Max Confidence:     {max_confidence:>12.4f}")
    print(f"  FG Avg Confidence:  {fg_confidence:>12.4f}")
    
    print("="*60 + "\n")

def save_visual_results(original_image: Any, predicted_mask: npt.NDArray[np.uint8], 
                       confidence_map: npt.NDArray[np.float32], save_path: str) -> None:
    """Save visual results as a composite image using OpenCV."""
    try:
        # Resize images to same dimensions
        h, w = predicted_mask.shape
        original_resized = cv2.resize(original_image, (w, h))
        
        # Convert mask to 3-channel for visualization
        mask_colored = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)
        
        # Create colored confidence map
        conf_normalized = (confidence_map * 255).astype(np.uint8)
        conf_colored = cv2.applyColorMap(conf_normalized, cv2.COLORMAP_JET)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)
        
        cv2.putText(original_resized, "Original", (10, 30), font, font_scale, color, thickness)
        cv2.putText(mask_colored, "Prediction", (10, 30), font, font_scale, color, thickness)
        cv2.putText(conf_colored, "Confidence", (10, 30), font, font_scale, color, thickness)
        
        # Concatenate horizontally
        composite = np.hstack([original_resized, mask_colored, conf_colored])
        
        # Save composite image
        cv2.imwrite(save_path, composite)
        print(f"  Visual results saved to: {save_path}")
        
    except Exception as e:
        print(f"  Warning: Failed to save visual results: {e}")


def inference(model: torch.nn.Module, image_path: str, device: torch.device,
              threshold: float = 0.5, visualize: bool = True, use_amp: bool = True) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """Run inference on a single image with AMP optimization."""
    try:
        # Preprocess image  
        image_tensor, original_image = preprocess_image(image_path)
        image_tensor = image_tensor.to(device, non_blocking=True)
        
        # Run inference with AMP
        use_amp = use_amp and device.type == 'cuda'
        with torch.no_grad():
            with autocast(enabled=use_amp):
                output = model(image_tensor)
            
            # Extract confidence map (avoid redundant operations)
            confidence_map = output.squeeze().cpu().numpy()
        
        # Postprocess
        binary_mask = postprocess_mask(output, threshold)
        
        # Display results if requested
        if visualize:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            print_inference_results(binary_mask, confidence_map, base_name)
            
            # Save visual composite
            save_path = f"{base_name}_composite.png"
            save_visual_results(original_image, binary_mask, confidence_map, save_path)
        
        return binary_mask, confidence_map
        
    except Exception as e:
        raise RuntimeError(f"Inference failed for {image_path}: {e}")


def batch_inference(model: torch.nn.Module, input_dir: str, output_dir: str, 
                   device: torch.device, threshold: float = 0.5, batch_size: int = 8, use_amp: bool = True):
    """Run true batch inference on all images in a directory with optimizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([f for f in os.listdir(input_dir) 
                          if f.lower().endswith(image_extensions)])
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images in batches of {batch_size}...")
    
    use_amp = use_amp and device.type == 'cuda'
    processed_count = 0
    error_count = 0
    
    try:
        # Process in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc="Batch progress"):
            batch_files = image_files[i:i + batch_size]
            batch_paths = [os.path.join(input_dir, f) for f in batch_files]
            
            try:
                # Preprocess batch
                batch_tensor, original_images = preprocess_batch(batch_paths)
                batch_tensor = batch_tensor.to(device, non_blocking=True)
                
                # Run inference on entire batch
                with torch.no_grad():
                    with autocast(enabled=use_amp):
                        outputs = model(batch_tensor)
                
                # Process each output
                for idx, (image_file, output) in enumerate(zip(batch_files, outputs)):
                    try:
                        # Postprocess
                        binary_mask = postprocess_mask(output.unsqueeze(0), threshold)
                        confidence_map = output.squeeze().cpu().numpy()
                        
                        # Save results
                        base_name = os.path.splitext(image_file)[0]
                        
                        # Save binary mask
                        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                        cv2.imwrite(mask_path, binary_mask)
                        
                        # Save confidence map
                        conf_path = os.path.join(output_dir, f"{base_name}_confidence.png")
                        conf_normalized = (confidence_map * 255).astype(np.uint8)
                        cv2.imwrite(conf_path, conf_normalized)
                        
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"\nError saving results for {image_file}: {e}")
                        error_count += 1
                
                # Clear cache periodically
                if device.type == 'cuda' and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\nError processing batch starting at {batch_files[0]}: {e}")
                error_count += len(batch_files)
    
    finally:
        # Cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\nBatch inference completed!")
    print(f"Successfully processed: {processed_count}/{len(image_files)} images")
    if error_count > 0:
        print(f"Errors encountered: {error_count} images")


def main():
    parser = argparse.ArgumentParser(description='U-Net VGG19 Inference with Optimizations')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary segmentation (0.0-1.0)')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch inference on directory')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for batch inference')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch.compile optimization')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    model = None
    try:
        # Load model with optimizations
        model = load_model(args.model, device, use_compile=not args.no_compile)
        
        use_amp = not args.no_amp
        if use_amp and device.type == 'cuda':
            print("Using automatic mixed precision (AMP)")
        
        if args.batch:
            # Batch inference
            if not os.path.isdir(args.input):
                print(f"Error: --batch requires a directory path. Got: {args.input}")
                sys.exit(1)
            batch_inference(model, args.input, args.output, device, args.threshold, 
                          args.batch_size, use_amp)
        else:
            # Single image inference
            if not os.path.isfile(args.input):
                print(f"Error: Input file not found: {args.input}")
                sys.exit(1)
            
            binary_mask, confidence_map = inference(
                model, args.input, device, args.threshold, visualize=True, use_amp=use_amp
            )
            
            # Save results
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            
            mask_path = os.path.join(args.output, f"{base_name}_mask.png")
            conf_path = os.path.join(args.output, f"{base_name}_confidence.png")
            
            cv2.imwrite(mask_path, binary_mask)
            conf_normalized = (confidence_map * 255).astype(np.uint8)
            cv2.imwrite(conf_path, conf_normalized)
            
            print(f"Results saved to {args.output}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup resources
        if model is not None:
            del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("\nResources cleaned up")


if __name__ == "__main__":
    main()