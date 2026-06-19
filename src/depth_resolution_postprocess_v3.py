#!/usr/bin/env python3
"""
Simplified depth-dependent Gaussian blur post-processing script.

This script applies a simple linear relationship between depth and Gaussian blur:
σ = 0.055986 * depth_μm

Based on USAF chart calibration data. The script processes alpha layers from
the forward model output directory and applies depth-dependent blur.

Usage:
python depth_resolution_postprocess_v3.py <output_dir>
"""

import os
import numpy as np
from PIL import Image
import argparse
import json
import glob
import re
from scipy.ndimage import gaussian_filter
import torch
import matplotlib.pyplot as plt

# Try to import OpenCV for faster Gaussian blur
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def apply_gaussian_blur_to_image(image, sigma):
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: Input image (numpy array or PIL Image)
        sigma: Gaussian blur sigma
        
    Returns:
        blurred_image: Blurred image (same type as input)
    """
    if sigma <= 0:
        return image
    
    if isinstance(image, Image.Image):
        # Convert PIL to numpy, blur, convert back
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            # RGB image - blur each channel
            blurred_array = np.zeros_like(img_array)
            for c in range(img_array.shape[2]):
                blurred_array[:, :, c] = gaussian_filter(img_array[:, :, c], sigma=sigma)
        else:
            # Grayscale image
            blurred_array = gaussian_filter(img_array, sigma=sigma)
        
        return Image.fromarray(blurred_array.astype(np.uint8))
    
    elif isinstance(image, np.ndarray):
        # Apply blur directly to numpy array
        if len(image.shape) == 3:
            # RGB image - blur each channel
            blurred = np.zeros_like(image)
            for c in range(image.shape[2]):
                blurred[:, :, c] = gaussian_filter(image[:, :, c], sigma=sigma)
        else:
            # Grayscale image
            blurred = gaussian_filter(image, sigma=sigma)
        return blurred
    
    else:
        raise ValueError("Unsupported image type")


def calculate_blur_sigma(depth_microns):
    """
    Calculate Gaussian blur sigma based on depth using linear relationship.
    
    Formula: σ = 0.055986 * depth_μm
    
    Args:
        depth_microns: Depth in microns
        
    Returns:
        sigma: Gaussian blur sigma in pixels
    """
    return 0.055986 * depth_microns


def simple_scatter(incident: torch.Tensor, scatter: float):
    """
    Apply simple scattering to incident light using Gaussian blur.
    
    Args:
        incident: Input light tensor [H, W]
        scatter: Scatter parameter (sigma for Gaussian blur)
        
    Returns:
        scattered: Scattered light tensor [H, W]
    """
    if scatter <= 0:
        return incident
    
    # Apply Gaussian blur using torch
    # Convert to numpy, apply scipy blur, convert back
    incident_np = incident.cpu().numpy()
    scattered_np = gaussian_filter(incident_np, sigma=scatter)
    return torch.tensor(scattered_np, dtype=incident.dtype, device=incident.device)


def ray_trace(incident: torch.Tensor, source_coords, D_layer):
    """
    Simple ray tracing simulation - for now just return the input.
    This would implement geometric ray tracing in a full implementation.
    
    Args:
        incident: Input light tensor [H, W]
        source_coords: Source coordinates
        D_layer: Layer distance
        
    Returns:
        traced: Ray traced light tensor [H, W]
    """
    # Simplified - just return input for now
    return incident


def forward_model_simplified(incident, alpha_layers, scatter=0.1, source_coords=None, D_layer=1.0):
    """
    Simplified forward model that combines incident light with alpha layers.
    
    Args:
        incident: Input light tensor of shape [3, H, W] for RGB
        alpha_layers: List of alpha layer tensors, each [H, W, 3] for RGB
        scatter: Scatter parameter
        source_coords: Coordinates of the light source (not used in simplified version)
        D_layer: Distance between layers
        
    Returns:
        final_output: Output RGB image of shape [3, H, W]
    """
    # Get number of channels (should be 3 for RGB)
    n_channels = incident.shape[0]
    num_layers = len(alpha_layers)
    
    # Initialize output placeholder for all channels
    output_channels = []
    
    # Process each color channel independently
    for c in range(n_channels):
        channel_incident = incident[c]  # Get single color channel [H, W]
        
        for layer_idx in range(num_layers):
            # Apply scattering
            channel_incident = simple_scatter(channel_incident, scatter)
            
            # Apply the alpha value for this channel and layer
            # alpha_layers[layer_idx] is [H, W, 3], we want channel c
            alpha_channel = alpha_layers[layer_idx][:, :, c]  # [H, W]
            channel_incident = channel_incident * alpha_channel
            
            # Apply ray tracing (simplified - just pass through)
            channel_incident = ray_trace(channel_incident, source_coords, D_layer)
            
        output_channels.append(channel_incident)  # Final output for this channel
    
    # Stack the output channels to form RGB image [3, H, W]
    final_output = torch.stack(output_channels)
    
    return final_output


def load_incident_light_images(incident_dir):
    """
    Load incident light images from directory with alpha removal.
    
    Args:
        incident_dir: Directory containing incident light images
        
    Returns:
        incident_images: List of loaded incident light images as tensors
        incident_filenames: List of corresponding filenames
    """
    incident_files = glob.glob(os.path.join(incident_dir, "*.png"))
    incident_files = sorted(incident_files, key=lambda x: natural_sort_key(os.path.basename(x)))
    
    incident_images = []
    incident_filenames = []
    
    for filepath in incident_files:
        try:
            img = Image.open(filepath)
            
            # Remove alpha channel to prevent checker patterns
            img = remove_alpha_channel(img)
            
            img_array = np.array(img)
            
            # Convert to tensor and ensure RGB format [3, H, W]
            if len(img_array.shape) == 3:
                # RGB image [H, W, 3] -> [3, H, W]
                img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            else:
                # Grayscale -> RGB
                img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0
                img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # [H, W] -> [3, H, W]
            
            incident_images.append(img_tensor)
            incident_filenames.append(os.path.basename(filepath))
        except Exception as e:
            print(f"Warning: Could not load incident light {filepath}: {e}")
            continue
    
    print(f"Successfully loaded {len(incident_images)} incident light images from {incident_dir}")
    return incident_images, incident_filenames


def load_alpha_layers_as_tensors(alpha_dir):
    """
    Load alpha layer images and convert to tensors with alpha removal.
    
    Args:
        alpha_dir: Directory containing alpha layer images
        
    Returns:
        alpha_tensors: List of alpha layer tensors, each [H, W, 3]
        alpha_filenames: List of corresponding filenames
    """
    alpha_files = glob.glob(os.path.join(alpha_dir, "*.png"))
    alpha_files = sorted(alpha_files, key=lambda x: natural_sort_key(os.path.basename(x)))
    
    alpha_tensors = []
    alpha_filenames = []
    
    for filepath in alpha_files:
        try:
            img = Image.open(filepath)
            
            # Remove alpha channel to prevent checker patterns
            img = remove_alpha_channel(img)
            
            img_array = np.array(img)
            
            # Convert to tensor and ensure format [H, W, 3]
            if len(img_array.shape) == 3:
                # RGB image [H, W, 3]
                img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0
            else:
                # Grayscale -> RGB [H, W] -> [H, W, 3]
                img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0
                img_tensor = img_tensor.unsqueeze(2).repeat(1, 1, 3)
            
            alpha_tensors.append(img_tensor)
            alpha_filenames.append(os.path.basename(filepath))
        except Exception as e:
            print(f"Warning: Could not load alpha layer {filepath}: {e}")
            continue
    
    print(f"Successfully loaded {len(alpha_tensors)} alpha layer tensors from {alpha_dir}")
    return alpha_tensors, alpha_filenames


def load_config_from_output_dir(output_dir):
    """
    Load configuration from JSON file in the output directory.
    
    Args:
        output_dir: Output directory that should contain a config JSON file
        
    Returns:
        config: Configuration dictionary, or None if not found
    """
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    if not json_files:
        print(f"Warning: No JSON config file found in {output_dir}")
        return None
    
    config_path = json_files[0]
    print(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return None


def extract_parameters_from_config(config):
    """
    Extract sample depth from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        sample_depth_microns: Sample depth in microns
    """
    try:
        sample_z_um = config['physical_parameters']['sample_z_um']
        print(f"Extracted sample depth: {sample_z_um} μm")
        return sample_z_um
    except KeyError as e:
        print(f"Warning: Could not extract sample depth from config, missing key: {e}")
        print("Using default value: 100.0 μm")
        return 100.0


def natural_sort_key(text):
    """
    Convert a string into a list of string and number chunks for natural sorting.
    "alpha2.png" -> ['alpha', 2, '.png']
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    
    return [tryint(c) for c in re.split('([0-9]+)', text)]


def remove_alpha_channel(img):
    """
    Remove alpha channel from PIL image and ensure no transparency.
    
    Args:
        img: PIL Image that may have alpha channel
        
    Returns:
        PIL Image without alpha channel (RGB or L mode)
    """
    if img.mode in ('RGBA', 'LA'):
        # Convert RGBA to RGB by compositing over white background
        background = Image.new('RGB' if img.mode == 'RGBA' else 'L', img.size, (255, 255, 255) if img.mode == 'RGBA' else 255)
        if img.mode == 'RGBA':
            img = Image.alpha_composite(background.convert('RGBA'), img).convert('RGB')
        else:  # LA mode
            # For grayscale with alpha, blend with white
            alpha = img.getchannel('A')
            gray = img.getchannel('L')
            # Blend: result = gray * alpha/255 + white * (1 - alpha/255)
            alpha_norm = np.array(alpha, dtype=np.float32) / 255.0
            gray_arr = np.array(gray, dtype=np.float32)
            result_arr = gray_arr * alpha_norm + 255 * (1 - alpha_norm)
            img = Image.fromarray(result_arr.astype(np.uint8), mode='L')
    elif img.mode == 'P':
        # Convert palette mode to RGB to avoid transparency issues
        img = img.convert('RGB')
    
    return img


def load_layer_images(layer_dir, layer_pattern="*.png"):
    """
    Load all layer images from a directory with natural sorting and alpha removal.
    
    Args:
        layer_dir: Directory containing layer images
        layer_pattern: Glob pattern for layer files
        
    Returns:
        images: List of loaded images
        filenames: List of corresponding filenames
    """
    layer_files = glob.glob(os.path.join(layer_dir, layer_pattern))
    # Apply natural sorting
    layer_files = sorted(layer_files, key=lambda x: natural_sort_key(os.path.basename(x)))
    
    images = []
    filenames = []
    
    for filepath in layer_files:
        try:
            img = Image.open(filepath)
            img.load()  # Load the image data to catch corrupted files
            
            # Remove alpha channel to prevent checker patterns
            img = remove_alpha_channel(img)
            
            images.append(img)
            filenames.append(os.path.basename(filepath))
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            continue
    
    print(f"Successfully loaded {len(images)} out of {len(layer_files)} images from {layer_dir}")
    return images, filenames


def process_alpha_layers(output_dir, sample_depth_microns):
    """
    Process alpha layer images by applying depth-dependent Gaussian blur.
    
    Args:
        output_dir: Output directory containing alpha_layers/ subdirectory
        sample_depth_microns: Total sample depth in microns
        
    Returns:
        Number of processed images
    """
    alpha_dir = os.path.join(output_dir, 'alpha_layers')
    
    if not os.path.exists(alpha_dir):
        print(f"Error: alpha_layers directory not found: {alpha_dir}")
        return 0
    
    # Create output directory for blurred results
    blurred_alpha_dir = os.path.join(output_dir, 'blurred_alpha_layers')
    os.makedirs(blurred_alpha_dir, exist_ok=True)
    print(f"Created output directory: {blurred_alpha_dir}")
    
    # Load alpha layer images
    alpha_images, alpha_files = load_layer_images(alpha_dir, "*.png")
    
    if not alpha_images:
        print("No alpha layer images found to process")
        return 0
    
    processed_count = 0
    
    for i, (img, filename) in enumerate(zip(alpha_images, alpha_files)):
        # Calculate depth for this layer 
        # Layer 0 is deepest (gets most blur), last layer is surface (gets least blur)
        layer_depth = ((len(alpha_images) - 1 - i) / max(len(alpha_images) - 1, 1)) * sample_depth_microns
        
        # Calculate blur sigma using linear formula
        sigma = calculate_blur_sigma(layer_depth)
        
        print(f"Processing {filename}: layer={i}, depth={layer_depth:.1f}μm, σ={sigma:.3f}")
        
        # Apply Gaussian blur
        blurred_image = apply_gaussian_blur_to_image(img, sigma)
        
        # Save the blurred image
        output_path = os.path.join(blurred_alpha_dir, filename)
        blurred_image.save(output_path)
        
        processed_count += 1
    
    print(f"Successfully processed {processed_count} alpha layer images")
    return processed_count


def generate_predicted_images(output_dir):
    """
    Generate predicted images by combining blurred alpha layers with incident light.
    
    Args:
        output_dir: Output directory containing blurred_alpha_layers/ and incident/ subdirectories
        
    Returns:
        Number of generated predicted images
    """
    incident_dir = os.path.join(output_dir, 'incident')
    blurred_alpha_dir = os.path.join(output_dir, 'blurred_alpha_layers')
    
    if not os.path.exists(incident_dir):
        print(f"Error: incident directory not found: {incident_dir}")
        return 0
        
    if not os.path.exists(blurred_alpha_dir):
        print(f"Error: blurred_alpha_layers directory not found: {blurred_alpha_dir}")
        return 0
    
    # Create output directory for blurred predicted results
    blurred_predicted_dir = os.path.join(output_dir, 'blurred_predicted')
    os.makedirs(blurred_predicted_dir, exist_ok=True)
    print(f"Created output directory: {blurred_predicted_dir}")
    
    # Load incident light images
    incident_images, incident_filenames = load_incident_light_images(incident_dir)
    
    # Load blurred alpha layers
    alpha_tensors, alpha_filenames = load_alpha_layers_as_tensors(blurred_alpha_dir)
    
    if not incident_images:
        print("No incident light images found")
        return 0
        
    if not alpha_tensors:
        print("No alpha layer images found")
        return 0
    
    print(f"Found {len(incident_images)} incident light images and {len(alpha_tensors)} alpha layers")
    
    generated_count = 0
    
    # Generate predicted image for each incident light
    for i, (incident_tensor, incident_filename) in enumerate(zip(incident_images, incident_filenames)):
        print(f"Processing incident light {incident_filename}...")
        
        # Apply forward model to combine incident light with all alpha layers
        predicted_tensor = forward_model_simplified(
            incident=incident_tensor,      # [3, H, W]
            alpha_layers=alpha_tensors,    # List of [H, W, 3]
            scatter=0.1,                   # Small scatter parameter
            source_coords=None,
            D_layer=1.0
        )
        
        # Convert tensor back to image format [H, W, 3]
        predicted_np = predicted_tensor.cpu().numpy().transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        predicted_np = np.clip(predicted_np, 0, 1)  # Ensure values are in [0, 1]
        
        # Save the predicted image
        output_filename = incident_filename.replace('.png', '_predicted.png')
        output_path = os.path.join(blurred_predicted_dir, output_filename)
        plt.imsave(output_path, predicted_np)
        
        print(f"Generated predicted image: {output_filename}")
        generated_count += 1
    
    print(f"Successfully generated {generated_count} predicted images")
    return generated_count


def main():
    parser = argparse.ArgumentParser(description='Apply linear depth-dependent blur to alpha layers and generate predicted images')
    parser.add_argument('--sample_depth', type=float, default=None,
                       help='Total sample depth in microns (if not provided, will be read from config)')
    
    args = parser.parse_args()
    
    # Hard-coded output directory
    output_dir = "/home/al/Documents/Motion_Detector/high_quality_results/output_20250814_tartrazine_transmission"
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}")
        return 1
    
    # Load configuration from output directory
    config = load_config_from_output_dir(output_dir)
    
    # Extract sample depth from config or use command line argument
    if config:
        config_sample_depth = extract_parameters_from_config(config)
    else:
        config_sample_depth = 100.0  # Default if no config
    
    # Use command line argument if provided, otherwise use config value
    sample_depth = args.sample_depth if args.sample_depth is not None else config_sample_depth
    
    print(f"Processing directory: {output_dir}")
    print(f"Sample depth: {sample_depth} μm")
    print(f"Using linear formula: σ = 0.055986 × depth_μm")
    print()
    
    # Process the alpha layers (blur them)
    processed_count = process_alpha_layers(output_dir, sample_depth)
    
    # Generate predicted images automatically
    print("\nGenerating predicted images...")
    generated_count = generate_predicted_images(output_dir)
    
    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Alpha layer images processed: {processed_count}")
    print(f"Predicted images generated: {generated_count}")
    
    return 0


if __name__ == "__main__":
    exit(main())