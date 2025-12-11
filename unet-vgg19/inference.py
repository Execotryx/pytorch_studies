import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from unet_vgg19 import UNetVGG19


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    config = checkpoint.get('config', {'num_classes': 1})
    
    # Create model
    model = UNetVGG19(num_classes=config['num_classes'], pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Best validation IoU: {checkpoint.get('best_val_iou', 'N/A')}")
    
    return model


def preprocess_image(image_path: str, image_size: tuple = (256, 256)) -> tuple:
    """Preprocess input image for inference."""
    # Load image using OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image_resized = cv2.resize(image_rgb, image_size)
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image_rgb


def postprocess_mask(mask_tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Postprocess model output to binary mask."""
    # Remove batch dimension and convert to numpy
    mask = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    return binary_mask


def visualize_results(original_image: np.ndarray, predicted_mask: np.ndarray, 
                     confidence_map: np.ndarray, save_path: str = None) -> None:
    """Visualize inference results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(predicted_mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    # Confidence map
    im = axes[2].imshow(confidence_map, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('Confidence Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    
    plt.show()


def inference(model: torch.nn.Module, image_path: str, device: torch.device,
              threshold: float = 0.5, visualize: bool = True) -> tuple:
    """Run inference on a single image."""
    # Preprocess image  
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        
        # If output is sigmoid activated, no need to apply sigmoid again
        # If not, you might need: output = torch.sigmoid(output)
        confidence_map = output.squeeze(0).squeeze(0).cpu().numpy()
    
    # Postprocess
    binary_mask = postprocess_mask(output, threshold)
    
    # Visualize if requested
    if visualize:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f"{base_name}_results.png"
        visualize_results(original_image, binary_mask, confidence_map, save_path)
    
    return binary_mask, confidence_map


def batch_inference(model: torch.nn.Module, input_dir: str, output_dir: str, 
                   device: torch.device, threshold: float = 0.5):
    """Run inference on all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    
    print(f"Processing {len(image_files)} images...")
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        
        try:
            # Run inference
            binary_mask, confidence_map = inference(
                model, image_path, device, threshold, visualize=False
            )
            
            # Save results
            base_name = os.path.splitext(image_file)[0]
            
            # Save binary mask
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, binary_mask)
            
            # Save confidence map
            conf_path = os.path.join(output_dir, f"{base_name}_confidence.png")
            conf_normalized = (confidence_map * 255).astype(np.uint8)
            cv2.imwrite(conf_path, conf_normalized)
            
            print(f"Processed: {image_file}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    print("Batch inference completed!")


def main():
    parser = argparse.ArgumentParser(description='U-Net VGG19 Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary segmentation')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch inference on directory')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    
    if args.batch:
        # Batch inference
        batch_inference(model, args.input, args.output, device, args.threshold)
    else:
        # Single image inference
        binary_mask, confidence_map = inference(
            model, args.input, device, args.threshold, visualize=True
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


if __name__ == "__main__":
    main()