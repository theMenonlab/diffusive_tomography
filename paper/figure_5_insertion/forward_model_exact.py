import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam
from PIL import Image
import torchvision.transforms as T
from torch.optim.lr_scheduler import LambdaLR
import os
import re
import glob
import math
import json
import sys
from matplotlib.widgets import Slider, Button, RadioButtons

# Import fourier resolution analysis
try:
    from fourier_resolution import analyze_alpha_layers_z_resolution
    FOURIER_ANALYSIS_AVAILABLE = True
except ImportError:
    print("Warning: fourier_resolution.py not found. Z-axis resolution analysis will be skipped.")
    FOURIER_ANALYSIS_AVAILABLE = False


def load_config(config_path=None):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file. If None, tries to use command line argument.
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            # Default to Amazon microscope config
            config_path = 'config_amazon_microscope.json'

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Please provide a valid config file path as command line argument.")
        print("Example: python forward_model_rbg_20250813_json_input.py config_amazon.json")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON configuration: {e}")
        sys.exit(1)


def forward_model_end(incident, alpha_t, scatter, source_coords, D_layer):
    """
    Forward model that supports RGB (3-channel) images
    
    Args:
        incident: Input light tensor of shape [3, H, W] for RGB
        alpha_t: Attenuation parameters of shape [H, W, num_layers, 3] for RGB
        scatter: Scatter parameter
        source_coords: Coordinates of the light source
        D_layer: Distance between layers
        
    Returns:
        final_output: Output RGB image of shape [3, H, W]
        layer_outputs: List of RGB layer outputs, each [3, H, W]
    """
    # Get number of channels (should be 3 for RGB)
    n_channels = incident.shape[0]
    layer_outputs = []
    
    # constrain alpha
    alpha_t = alpha_t.clamp(0, 1)
    
    num_layers = alpha_t.shape[-2]  # Changed index due to new dimension for RGB
    
    # Initialize output placeholder for all channels
    output_channels = []
    all_layer_outputs = []
    
    # Process each color channel independently
    for c in range(n_channels):
        channel_incident = incident[c]  # Get single color channel
        channel_layer_outputs = []
        
        for layer_idx in range(num_layers):
            channel_incident = simple_scatter(channel_incident, scatter)
            # Apply the alpha value for this channel and layer
            channel_incident = channel_incident * alpha_t[..., layer_idx, c]
            
            channel_incident = ray_trace(channel_incident, source_coords, D_layer)
            channel_layer_outputs.append(channel_incident.clone())  # store a copy
            
        output_channels.append(channel_incident)  # Final output for this channel
        all_layer_outputs.append(channel_layer_outputs)
    
    # Combine outputs for all layers across channels
    for i in range(num_layers):
        layer_output = torch.stack([all_layer_outputs[c][i] for c in range(n_channels)])
        layer_outputs.append(layer_output)
    
    # Stack the output channels to form RGB image
    final_output = torch.stack(output_channels)
    
    return final_output, layer_outputs

def gaussian_kernel(size: int, sigma: float):
    """Create a 1D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel

def simple_scatter(incident: torch.Tensor, scatter: float):
    """
    Apply a Gaussian blur to a PyTorch tensor.
    Works with both single channel and RGB inputs.
    """
    if scatter <= 0:
        return incident  # No blur for zero or negative scatter

    # Determine the kernel size (at least 3, odd, and proportional to scatter)
    kernel_size = max(3, int(scatter * 6) | 1)
    kernel = gaussian_kernel(kernel_size, scatter).to(incident.device)

    # Create 2D Gaussian kernel by outer product
    kernel2d = kernel[:, None] * kernel[None, :]
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, K, K)

    # Add channel and batch dimensions if necessary
    if incident.ndim == 2:  # H x W
        incident = incident.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    elif incident.ndim == 3:  # C x H x W
        incident = incident.unsqueeze(0)  # Shape: (1, C, H, W)

    # Apply Gaussian blur using depthwise convolution
    blurred = F.conv2d(incident, kernel2d, padding=kernel_size // 2, groups=incident.size(1))

    # Squeeze out added dimensions if necessary
    if blurred.ndim == 4 and blurred.size(0) == 1 and blurred.size(1) == 1:
        return blurred.squeeze(0).squeeze(0)  # Return 2D tensor
    elif blurred.ndim == 4 and blurred.size(0) == 1:
        return blurred.squeeze(0)  # Return 3D tensor

    return blurred

def linear_lr(epoch, num_epochs=100):
    # Return a factor between 1.0 (start) and 0.001 (end)
    return (1.0 - epoch / (num_epochs)) * 0.999 + 0.001

def numerical_sort(value):
    # This regular expression finds all the digits in the filename and returns them.
    parts = re.compile(r'(\d+)').split(value)
    parts[1::2] = map(int, parts[1::2])  # Convert the extracted digits to integers for proper sorting
    return parts

def load_images(folder_path):
    images = []

    transform = T.Compose([
        T.ToTensor()         # Convert image to a tensor with values in [0, 1]
    ])
    image_files = glob.glob(os.path.join(folder_path, '*'))
    # Sort the files using the custom numerical_sort function
    image_files.sort(key=numerical_sort)
    for file_name in image_files:  # Sort to maintain order
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Check for valid image files
            img = Image.open(file_name).convert('RGB')  # Convert to grayscale
            images.append(transform(img))
    return torch.stack(images).squeeze(1)  # Stack tensors into a single tensor [num_images, channels, height, width]

def crop_to_center(image, target_height, target_width):
    """
    Crop the image to the target height and width around its center.
    Works with both single channel [H,W] and RGB [3,H,W] images.
    """
    if image.ndim == 2:  # Single channel
        h, w = image.shape
        top = (h - target_height) // 2
        left = (w - target_width) // 2
        return image[top:top + target_height, left:left + target_width]
    else:  # RGB
        _, h, w = image.shape
        top = (h - target_height) // 2
        left = (w - target_width) // 2
        return image[:, top:top + target_height, left:left + target_width]

def loss_cropper(predicted, layer_outputs, target, incident):
    """
    Crop all inputs to the size of the target.
    Supports both single channel and RGB images.
    """
    if target.ndim == 2:  # Single channel
        target_height, target_width = target.shape
    else:  # RGB
        _, target_height, target_width = target.shape
    
    # Crop each input
    cropped_predicted = crop_to_center(predicted, target_height, target_width)
    cropped_layer_outputs = [crop_to_center(layer, target_height, target_width) for layer in layer_outputs]
    cropped_incident = crop_to_center(incident, target_height, target_width)
    
    return cropped_predicted, cropped_layer_outputs, target, cropped_incident

def ray_trace(incident, source_coords, D_layer):
    # incident: 2D tensor [H, W]
    # source_coords: [x_src, y_src, z_src]
    # D_layer: distance along z to "layer"
    # Works for both single channel and RGB (call separately for each channel)

    H, W = incident.shape
    y_src, x_src, z_src = source_coords

    # Create 1D coordinate arrays for row (vertical) and column (horizontal)
    # Make sure these are on the same device as incident
    device = incident.device
    row_in = torch.arange(H, dtype=torch.float32, device=device)
    col_in = torch.arange(W, dtype=torch.float32, device=device)

    # Replicate original shift logic, keeping them as floats
    #x_out = (D_layer / -z_src) * -(row_in - x_src)  # shift in rows (vertical)
    #y_out = (D_layer / -z_src) * -(col_in - y_src)  # shift in columns (horizontal)
    x_out = (D_layer / z_src) * (row_in - x_src)  # shift in rows (vertical)
    y_out = (D_layer / z_src) * (col_in - y_src)  # shift in columns (horizontal)

    # Create a meshgrid of output pixel indices (rows -> xx_out, cols -> yy_out)
    xx_out, yy_out = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # Compute sub-pixel source coordinates for each output pixel
    # row_in_subpix => which row in the input do we sample?
    # col_in_subpix => which col in the input do we sample?
    row_in_subpix = xx_out + x_out.view(-1, 1)  # Broadcasting x_out across columns
    col_in_subpix = yy_out + y_out.view(1, -1)  # Broadcasting y_out across rows

    # Convert to normalized coordinates in [-1, 1] for grid_sample
    # grid[..., 0] = x-coord (horizontal), grid[..., 1] = y-coord (vertical).
    x_norm = 2.0 * (col_in_subpix / (W - 1)) - 1.0  # horizontal
    y_norm = 2.0 * (row_in_subpix / (H - 1)) - 1.0  # vertical

    # Combine into a single grid of shape [1, H, W, 2]
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

    # Reshape 'incident' to [N=1, C=1, H, W]
    incident = incident.unsqueeze(0).unsqueeze(0)

    # Sample with bilinear interpolation
    warped = F.grid_sample(
        incident,
        grid,
        mode='bilinear',
        padding_mode='reflection',
        align_corners=False
    )

    # Remove extra dimensions
    return warped.squeeze(0).squeeze(0)

def generate_synthetic_incident_light(
    image_size,  # e.g. 608
    source_coord,  # (x_s, y_s, z_s)
    margin,
    NA=0.39,
    intensity_mode='inverse_square',  
    blur_sigma=50,
    device='cpu',
    num_channels=3  # Default to 3 for RGB
):
    """
    Generate a illumination pattern at z=0 for a multimodal fiber source
    with support for RGB (3 channels)
    """
    x_s, y_s, z_s = source_coord
    
    H_expanded = image_size + 2 * margin
    W_expanded = image_size + 2 * margin
    
    # Build coordinate grid for the expanded plane
    yv, xv = torch.meshgrid(
        torch.arange(H_expanded, dtype=torch.float32, device=device),
        torch.arange(W_expanded, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # Compute world coordinates
    x_world = xv - margin
    y_world = yv - margin
    z_world = torch.zeros_like(x_world)  # z=0 plane

    # Distance from source
    dx = (x_world - x_s)
    dy = (y_world - y_s)
    dz = (z_world - z_s)
    r_squared = dx*dx + dy*dy + dz*dz + 1e-8  # small eps to avoid divide-by-zero
    r = torch.sqrt(r_squared)

    # Fiber divergence cone (multimodal uniform intensity model)
    divergence_angle = torch.asin(torch.tensor(NA, dtype=torch.float32, device=device))
    beam_radius_at_z = z_s * torch.tan(divergence_angle) # fiber diameter is small. can be simplified away

    # Create a binary mask for points within the cone radius at z
    radius_squared = dx * dx + dy * dy
    in_cone = radius_squared <= (beam_radius_at_z ** 2)
    
    # Intensity model
    if intensity_mode == 'inverse_square':
        # Using torch.div for better gradient stability
        incident_base = torch.div(torch.ones_like(r_squared), r_squared)
        max_intensity = 1.0 / (z_s**2)  # Maximum intensity at the source plane
    elif intensity_mode == 'gaussian':
        sigma = beam_radius_at_z/2
        sigma_squared = sigma * sigma
        r2_2d = dx*dx + dy*dy
        incident_base = torch.exp(-0.5 * r2_2d / sigma_squared)
        incident_base /= incident_base.max().clamp_min(1e-8)
        max_intensity = 1.0
    elif intensity_mode == 'multimodal':
        # Default: multimodal uniform in cone
        incident_base = in_cone.to(torch.float32)
        max_intensity = 1.0
    elif intensity_mode == 'multimodal_gaussian':
        # Default: multimodal uniform in cone
        incident_multimodal = in_cone.to(torch.float32)

        sigma = image_size/2  # or your desired sigma in pixels
        sigma_squared = sigma * sigma
        r2_2d = dx*dx + dy*dy
        incident_gaussian = torch.exp(-0.5 * r2_2d / sigma_squared)
        incident_gaussian /= incident_gaussian.max().clamp_min(1e-8)

        incident_base = incident_multimodal + incident_gaussian
        max_intensity = 1 + 1  # Maximum intensity is 1 for Gaussian and 1 for multimodal
    else:
        # default uniform
        incident_base = torch.ones_like(r)
        max_intensity = 1.0

    # Normalize to peak intensity of 1
    incident_base = incident_base / max_intensity

    # Apply Gaussian blur if blur_sigma is specified
    if blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        # Convert to numpy for filtering
        if isinstance(incident_base, torch.Tensor):
            incident_numpy = incident_base.cpu().numpy()
            # Apply Gaussian blur
            incident_numpy = gaussian_filter(incident_numpy, sigma=blur_sigma)
            # Convert back to tensor
            incident_base = torch.from_numpy(incident_numpy).to(incident_base.device)
        else:
            incident_base = gaussian_filter(incident_base, sigma=blur_sigma)
    
    
    # Create a multi-channel version for RGB
    if num_channels == 3:  # RGB
        # Stack the same pattern three times for RGB
        # In a more sophisticated model, you might have different patterns for each channel
        incident = torch.stack([incident_base, incident_base, incident_base])
    else:
        incident = incident_base

    return incident  # shape = [3, H_expanded, W_expanded] for RGB or [H_expanded, W_expanded] for single channel

def crop_incident_to_final(incident_expanded, margin):
    """
    Utility to crop the center region from the expanded incident.
    Works with both single channel [H,W] and RGB [3,H,W] or [H,W,3] images.
    """
    if incident_expanded.ndim == 2:  # Single channel [H, W]
        H, W = incident_expanded.shape
        final_size = W - 2*margin
        cropped = incident_expanded[margin:margin+final_size, margin:margin+final_size]
    elif incident_expanded.shape[0] == 3 and incident_expanded.ndim == 3:  # RGB channels first [3, H, W]
        _, H, W = incident_expanded.shape
        final_size = W - 2*margin
        cropped = incident_expanded[:, margin:margin+final_size, margin:margin+final_size]
    elif incident_expanded.shape[2] == 3 and incident_expanded.ndim == 3:  # RGB channels last [H, W, 3]
        H, W, _ = incident_expanded.shape
        final_size = W - 2*margin
        cropped = incident_expanded[margin:margin+final_size, margin:margin+final_size, :]
    else:
        raise ValueError(f"Unexpected incident shape: {incident_expanded.shape}")
        
    return cropped

def save_error_map(predicted, ground_truth, output_path, title="Prediction Error", 
                   vmin=None, vmax=None, colormap='viridis'):
    """
    Save a 2D absolute error map with proper normalization and color bar.
    
    Args:
        predicted: Predicted image tensor [3, H, W] or numpy array [H, W, 3]
        ground_truth: Ground truth image tensor [3, H, W] or numpy array [H, W, 3]  
        output_path: Path to save the error map
        title: Title for the plot
        vmin, vmax: Optional min/max values for color scale normalization
        colormap: Matplotlib colormap name
    """
    # Convert tensors to numpy if needed
    if hasattr(predicted, 'cpu'):
        predicted = predicted.detach().cpu().numpy()
    if hasattr(ground_truth, 'cpu'):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    # Ensure both arrays have the same shape and format [H, W, 3]
    if predicted.ndim == 3 and predicted.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
        predicted = np.transpose(predicted, (1, 2, 0))
    if ground_truth.ndim == 3 and ground_truth.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
        ground_truth = np.transpose(ground_truth, (1, 2, 0))
    
    # Check that shapes match before calculating error
    if predicted.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs ground_truth {ground_truth.shape}")
    
    # Calculate absolute error per pixel (average across RGB channels)
    abs_error = np.abs(predicted - ground_truth)
    if abs_error.ndim == 3:  # Average across color channels
        error_map = np.mean(abs_error, axis=2)  # [H, W]
    else:
        error_map = abs_error
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine color scale limits
    if vmin is None:
        vmin = np.min(error_map)
    if vmax is None:
        vmax = np.max(error_map)
    
    # Create the error map plot
    im = ax.imshow(error_map, cmap=colormap, vmin=vmin, vmax=vmax, aspect='equal')
    
    # Add colorbar with proper labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Absolute Error', rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    # Add statistics text
    mean_error = np.mean(error_map)
    max_error = np.max(error_map)
    std_error = np.std(error_map)
    
    stats_text = f'Mean Error: {mean_error:.4f}\nMax Error: {max_error:.4f}\nStd Error: {std_error:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    return error_map, {'mean': mean_error, 'max': max_error, 'std': std_error}

def loss_function_alpha(predicted, layer_outputs, target, incident, alpha_torch):
    # Crop all inputs to the size of the target
    predicted, layer_outputs, target, incident = loss_cropper(predicted, layer_outputs, target, incident)
    
    # Compute main loss - works for both RGB and single channel
    final_loss = F.mse_loss(predicted, target)
    
    # Add pixel value consistency across layers with larger weight
    layer_pixel_consistency_loss = layer_pixel_balance(alpha_torch) * 5.0
    
    # Add variance of Laplacian consistency with larger weight
    var_lap_consistency_loss = variance_of_laplacian_consistency(alpha_torch) * 20.0
    
    # Return combined loss with larger weights for new components
    total_loss = final_loss + layer_pixel_consistency_loss * var_lap_consistency_loss
    #print(f'total_loss: {total_loss.item()}, final_loss: {final_loss.item()}, layer_pixel_consistency_loss: {layer_pixel_consistency_loss.item()}, var_lap_consistency_loss: {var_lap_consistency_loss.item()}')
    
    return total_loss

def layer_pixel_balance(alpha_torch):
    """
    Encourages the average pixel value of each layer to be similar.
    alpha_torch shape: [648, 648, 16, 3] for RGB
    """
    # Calculate mean pixel value for each layer, for each channel
    layer_means = torch.mean(alpha_torch, dim=(0, 1))  # Shape: [16, 3] for RGB
    
    # Calculate the global average across all layers for each channel
    global_mean = torch.mean(layer_means, dim=0)  # Shape: [3] for RGB
    
    # Use L1 loss instead of MSE for stronger gradients
    # For each channel separately
    consistency_loss = torch.mean(torch.abs(layer_means - global_mean.unsqueeze(0)))
    
    return consistency_loss

def variance_of_laplacian(img):
    """
    Calculate the variance of the Laplacian for a single layer image.
    Used as a measure of image focus/sharpness.
    Works with both single channel and RGB images (apply separately to each channel).
    """
    # Define Laplacian kernel
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32).reshape(1, 1, 3, 3)
    
    if torch.cuda.is_available():
        laplacian_kernel = laplacian_kernel.cuda()
    
    # Add batch and channel dimensions if needed
    if len(img.shape) == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = img.permute(2, 0, 1).unsqueeze(0)
    
    # Apply the Laplacian filter
    laplacian = F.conv2d(img, laplacian_kernel, padding=1)
    
    # Calculate the variance
    var = torch.var(laplacian)
    
    return var

def variance_of_laplacian_consistency(alpha_torch):
    """
    Encourages the variance of the Laplacian of each layer to be similar.
    alpha_torch shape: [648, 648, 16, 3] for RGB
    """
    # For RGB, we need to handle each channel separately
    num_channels = alpha_torch.shape[-1]
    var_lap_losses = []
    
    for c in range(num_channels):
        # Calculate variance of Laplacian for each layer in this channel
        var_laps = []
        for i in range(alpha_torch.shape[2]):  # For each layer
            layer = alpha_torch[:, :, i, c]
            var_lap = variance_of_laplacian(layer)
            var_laps.append(var_lap)
        
        var_laps = torch.stack(var_laps)  # Shape: [16]
         
        # Calculate the average variance
        avg_var_lap = torch.mean(var_laps)
        
        # Use L1 loss
        var_lap_loss = torch.mean(torch.abs(var_laps - avg_var_lap))
        var_lap_losses.append(var_lap_loss)
    
    # Average the loss across all channels
    return torch.mean(torch.stack(var_lap_losses))

def gen_source_coords(num_images, image_size, img_x_um, z_depth_mm, right_to_left, light_step, center_image=None, y_position=None):
    """
    Generate light source coordinates for diffusive tomography.
    
    Args:
        num_images: Number of images in the sequence
        image_size: Size of the images in pixels (assumes square)
        img_x_um: Field of view in microns
        z_depth_mm: Z depth of light source in mm (will be converted to negative pixels)
        right_to_left: Boolean, True for right-to-left movement, False for left-to-right
        light_step: Step size between positions in mm
        center_image: Which image index has the centered light source (default: num_images//2)
        y_position: Y position in pixels (default: image_size/2)
    
    Returns:
        source_coords: List of [x, y, z] coordinates for each image
    """
    # Set defaults if not provided
    if center_image is None:
        center_image = num_images // 2
    if y_position is None:
        y_position = image_size / 2
    
    # Convert z to pixels (negative depth)
    z = -z_depth_mm * 1000 * image_size / img_x_um

    # Set direction multiplier
    if right_to_left:
        direction = -1
    else:
        direction = 1

    source_coords = []
    for i in range(num_images):
        # Calculate x position relative to center image
        offset_images = i - center_image
        x_offset_mm = offset_images * light_step * direction
        x_offset_pixels = x_offset_mm * 1000 * image_size / img_x_um
        x = image_size / 2 + x_offset_pixels
        
        source_coords.append([x, y_position, z])
    
    return source_coords

def interactive_source_coords_gui(image_stack, image_size, num_images, config=None):
    """
    Interactive GUI for setting light source coordinates with real-time preview.
    
    Args:
        image_stack: Tensor of shape [num_images, 3, height, width] for RGB images
        image_size: Size of the images (assumes square images)
        num_images: Number of images in the stack
        config: Configuration dictionary with default values from JSON
        
    Returns:
        source_coords: List of [x, y, z] coordinates for each image
    """
    
    # Convert image stack to numpy for display (handle both tensor and numpy inputs)
    if hasattr(image_stack, 'cpu'):
        images_np = image_stack.cpu().numpy()
    else:
        images_np = image_stack
    
    # Initialize parameters with defaults from config or fallback values
    if config is not None:
        # Use values from config with fallbacks
        default_fov = config.get('physical_parameters', {}).get('img_x_um', 940.0)
        default_z_depth = config.get('light_source', {}).get('z_depth_mm', 5.8)
        default_y_pos = config.get('light_source', {}).get('y_position', image_size / 2)
        default_step = config.get('light_source', {}).get('light_step', 0.2)
        default_center = config.get('light_source', {}).get('center_image', num_images // 2)
        default_direction = -1 if config.get('light_source', {}).get('right_to_left', False) else 1
        default_intensity_mode = config.get('light_source', {}).get('intensity_mode', 'gaussian')
        default_na = config.get('light_source', {}).get('NA', 0.39)
        default_blur_sigma = config.get('light_source', {}).get('blur_sigma', 50)
        
        # Handle None values from config
        if default_y_pos is None:
            default_y_pos = image_size / 2
        if default_center is None:
            default_center = num_images // 2
    else:
        # Fallback defaults if no config provided
        default_fov = 940.0
        default_z_depth = 5.8
        default_y_pos = image_size / 2
        default_step = 0.2
        default_center = num_images // 2
        default_direction = 1
        default_intensity_mode = 'gaussian'
        default_na = 0.39
        default_blur_sigma = 50
    
    params = {
        'fov_microns': default_fov,        # Field of view in microns from config
        'z_depth_mm': default_z_depth,     # Z depth from config
        'y_position': default_y_pos,       # Y position from config
        'step_size_mm': default_step,      # Step size from config
        'center_image': default_center,    # Center image from config
        'current_image': 0,                # Currently displayed image
        'direction': default_direction,    # Direction from config (1 for left-to-right, -1 for right-to-left)
        'intensity_mode': default_intensity_mode,  # Light source intensity mode
        'NA': default_na,                  # Numerical aperture
        'blur_sigma': default_blur_sigma,  # Blur sigma
        'view_mode': 'sample',            # Current view mode: 'sample' or 'light'
        'confirmed': False                 # Whether user confirmed the settings
    }
    
    # Create the figure and subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Main image display
    ax_image = plt.subplot2grid((5, 5), (0, 0), colspan=3, rowspan=4)
    
    # Parameter display
    ax_params = plt.subplot2grid((5, 5), (0, 3), colspan=2)
    ax_params.axis('off')
    
    # Light source controls area
    ax_light_controls = plt.subplot2grid((5, 5), (1, 3), colspan=2)
    ax_light_controls.axis('off')
    
    # Sliders area
    slider_area = plt.subplot2grid((5, 5), (2, 3), colspan=2, rowspan=2)
    slider_area.axis('off')
    
    # Current coordinates display
    ax_coords = plt.subplot2grid((5, 5), (4, 0), colspan=3)
    ax_coords.axis('off')
    
    # Display initial image
    im = ax_image.imshow(np.transpose(images_np[0], (1, 2, 0)))
    ax_image.set_title('Image 0 - Use sliders to adjust light source position')
    ax_image.axis('on')
    
    # Add crosshair for light source position
    crosshair_v = ax_image.axvline(x=image_size/2, color='red', linewidth=2, alpha=0.7)
    crosshair_h = ax_image.axhline(y=image_size/2, color='red', linewidth=2, alpha=0.7)
    
    # View mode toggle button
    ax_view_toggle = plt.axes([0.72, 0.90, 0.12, 0.05])
    button_view_toggle = Button(ax_view_toggle, 'View: Light')  # Initial mode is 'sample', so show what you'll switch TO
    
    # Light source parameter controls
    ax_intensity_mode = plt.axes([0.72, 0.82, 0.12, 0.06])
    radio_intensity = RadioButtons(ax_intensity_mode, ('gaussian', 'inverse_square', 'multimodal', 'multimodal_gaussian'))
    # Set default based on config
    intensity_modes = ['gaussian', 'inverse_square', 'multimodal', 'multimodal_gaussian']
    if params['intensity_mode'] in intensity_modes:
        radio_intensity.set_active(intensity_modes.index(params['intensity_mode']))
    
    # Create sliders
    slider_height = 0.025
    slider_spacing = 0.035
    slider_left = 0.72
    slider_width = 0.12
    
    # Position sliders
    current_y = 0.75
    
    # FOV slider
    ax_fov = plt.axes([slider_left, current_y, slider_width, slider_height])
    slider_fov = Slider(ax_fov, 'FOV (μm)', 100, 2000, valinit=params['fov_microns'], valfmt='%.0f')
    current_y -= slider_spacing
    
    # Z depth slider
    ax_z = plt.axes([slider_left, current_y, slider_width, slider_height])
    slider_z = Slider(ax_z, 'Z depth (mm)', 0.5, 20.0, valinit=params['z_depth_mm'], valfmt='%.1f')
    current_y -= slider_spacing
    
    # Y position slider
    ax_y = plt.axes([slider_left, current_y, slider_width, slider_height])
    slider_y = Slider(ax_y, 'Y pos (px)', 0, image_size, valinit=params['y_position'], valfmt='%.0f')
    current_y -= slider_spacing
    
    # Step size slider
    ax_step = plt.axes([slider_left, current_y, slider_width, slider_height])
    slider_step = Slider(ax_step, 'Step (mm)', 0.05, 1.0, valinit=params['step_size_mm'], valfmt='%.2f')
    current_y -= slider_spacing
    
    # Center image slider
    ax_center = plt.axes([slider_left, current_y, slider_width, slider_height])
    slider_center = Slider(ax_center, 'Center img', 0, num_images-1, valinit=params['center_image'], valfmt='%.0f')
    current_y -= slider_spacing
    
    # Numerical Aperture slider
    ax_na = plt.axes([slider_left, current_y, slider_width, slider_height])
    slider_na = Slider(ax_na, 'NA', 0.1, 1.0, valinit=params['NA'], valfmt='%.2f')
    current_y -= slider_spacing
    
    # Blur sigma slider
    ax_blur = plt.axes([slider_left, current_y, slider_width, slider_height])
    slider_blur = Slider(ax_blur, 'Blur σ', 10, 200, valinit=params['blur_sigma'], valfmt='%.0f')
    current_y -= slider_spacing
    
    # Add extra spacing before direction controls
    current_y -= 0.04
    
    # Current image slider (big one)
    ax_current = plt.axes([0.1, 0.02, 0.5, 0.04])
    slider_current = Slider(ax_current, 'Image', 0, num_images-1, valinit=0, valfmt='%.0f')
    
    # Direction radio buttons
    ax_direction = plt.axes([slider_left, current_y, slider_width, 0.06])
    radio_direction = RadioButtons(ax_direction, ('Left→Right', 'Right→Left'))
    # Set default based on config direction
    radio_direction.set_active(0 if params['direction'] == 1 else 1)
    current_y -= 0.08
    
    # Confirm button
    ax_confirm = plt.axes([slider_left, current_y, slider_width, 0.05])
    button_confirm = Button(ax_confirm, 'Confirm Settings')
    
    def calculate_source_coords():
        """Calculate source coordinates based on current parameters"""
        fov_microns = params['fov_microns']
        z_depth_mm = params['z_depth_mm']
        y_pos = params['y_position']
        step_mm = params['step_size_mm']
        center_img = params['center_image']
        direction = params['direction']
        
        # Convert z to pixels (negative depth)
        z = -z_depth_mm * 1000 * image_size / fov_microns
        
        source_coords = []
        for i in range(num_images):
            # Calculate x position relative to center image
            offset_images = i - center_img
            x_offset_mm = offset_images * step_mm * direction
            x_offset_pixels = x_offset_mm * 1000 * image_size / fov_microns
            x = image_size / 2 + x_offset_pixels
            
            source_coords.append([x, y_pos, z])
        
        return source_coords
    
    def generate_current_light():
        """Generate synthetic incident light for current image and parameters"""
        coords = calculate_source_coords()
        current_img = int(params['current_image'])
        source_coord = coords[current_img]
        
        # Generate synthetic light
        synthetic_light = generate_synthetic_incident_light(
            image_size=image_size,
            source_coord=source_coord,
            margin=20,  # Small margin for generation
            NA=params['NA'],
            intensity_mode=params['intensity_mode'],
            blur_sigma=params['blur_sigma'],
            device='cpu',
            num_channels=3
        )
        
        # Crop to final size
        if synthetic_light.ndim == 3 and synthetic_light.shape[0] == 3:
            # RGB format [3, H, W]
            margin = (synthetic_light.shape[1] - image_size) // 2
            cropped = synthetic_light[:, margin:margin+image_size, margin:margin+image_size]
            # Convert to display format [H, W, 3]
            display_light = np.transpose(cropped.numpy(), (1, 2, 0))
        else:
            # Single channel
            margin = (synthetic_light.shape[0] - image_size) // 2
            cropped = synthetic_light[margin:margin+image_size, margin:margin+image_size]
            # Convert to RGB for display
            display_light = np.stack([cropped.numpy()] * 3, axis=2)
        
        return display_light
    
    def update_display():
        """Update the image display and crosshair"""
        current_img = int(params['current_image'])
        
        # Calculate current light position
        coords = calculate_source_coords()
        x_pos = coords[current_img][0]
        y_pos = coords[current_img][1]
        
        # Update image based on view mode
        if params['view_mode'] == 'sample':
            # Show sample image
            im.set_array(np.transpose(images_np[current_img], (1, 2, 0)))
            ax_image.set_title(f'Sample Image {current_img} - Light source position')
        else:  # 'light' mode
            # Show synthetic incident light
            try:
                light_image = generate_current_light()
                im.set_array(light_image)
                ax_image.set_title(f'Synthetic Incident Light {current_img} - {params["intensity_mode"]}')
            except Exception as e:
                print(f"Error generating light: {e}")
                # Fallback to sample image
                im.set_array(np.transpose(images_np[current_img], (1, 2, 0)))
                ax_image.set_title(f'Sample Image {current_img} - (Light generation failed)')
        
        # Update crosshair
        crosshair_v.set_xdata([x_pos, x_pos])
        crosshair_h.set_ydata([y_pos, y_pos])
        
        # Update parameter display
        ax_params.clear()
        ax_params.axis('off')
        param_text = f"""Parameters:
FOV: {params['fov_microns']:.0f} μm
Z depth: {params['z_depth_mm']:.1f} mm
Y position: {params['y_position']:.0f} px
Step size: {params['step_size_mm']:.2f} mm
Center image: {params['center_image']:.0f}
Direction: {'L→R' if params['direction'] == 1 else 'R→L'}

Light Source:
Mode: {params['intensity_mode']}
NA: {params['NA']:.2f}
Blur σ: {params['blur_sigma']:.0f}

Position (img {current_img}):
X: {x_pos:.1f} px
Y: {y_pos:.1f} px
Z: {coords[current_img][2]:.1f} px"""
        
        ax_params.text(0.005, 0.95, param_text, transform=ax_params.transAxes, 
                      verticalalignment='top', fontsize=9, fontfamily='monospace')
        
        # Update coordinates display
        ax_coords.clear()
        ax_coords.axis('off')
        view_text = f"View: {params['view_mode'].title()} | "
        coord_text = f"Source coordinates for image {current_img}: X={x_pos:.1f}, Y={y_pos:.1f}, Z={coords[current_img][2]:.1f} pixels"
        ax_coords.text(0.5, 0.5, view_text + coord_text, transform=ax_coords.transAxes, 
                      ha='center', va='center', fontsize=11, weight='bold')
        
        fig.canvas.draw()
    
    def on_fov_change(val):
        params['fov_microns'] = val
        update_display()
    
    def on_z_change(val):
        params['z_depth_mm'] = val
        update_display()
    
    def on_y_change(val):
        params['y_position'] = val
        update_display()
    
    def on_step_change(val):
        params['step_size_mm'] = val
        update_display()
    
    def on_center_change(val):
        params['center_image'] = int(val)
        update_display()
    
    def on_current_change(val):
        params['current_image'] = int(val)
        update_display()
    
    def on_direction_change(label):
        params['direction'] = 1 if label == 'Left→Right' else -1
        update_display()
    
    def on_na_change(val):
        params['NA'] = val
        update_display()
    
    def on_blur_change(val):
        params['blur_sigma'] = val
        update_display()
    
    def on_intensity_mode_change(label):
        params['intensity_mode'] = label
        update_display()
    
    def on_view_toggle(event):
        if params['view_mode'] == 'sample':
            params['view_mode'] = 'light'
            button_view_toggle.label.set_text('View: Sample')  # Show what you'll switch TO
        else:
            params['view_mode'] = 'sample' 
            button_view_toggle.label.set_text('View: Light')   # Show what you'll switch TO
        update_display()
    
    def on_confirm(event):
        params['confirmed'] = True
        plt.close()
    
    # Connect sliders to update functions
    slider_fov.on_changed(on_fov_change)
    slider_z.on_changed(on_z_change)
    slider_y.on_changed(on_y_change)
    slider_step.on_changed(on_step_change)
    slider_center.on_changed(on_center_change)
    slider_current.on_changed(on_current_change)
    slider_na.on_changed(on_na_change)
    slider_blur.on_changed(on_blur_change)
    radio_direction.on_clicked(on_direction_change)
    radio_intensity.on_clicked(on_intensity_mode_change)
    button_view_toggle.on_clicked(on_view_toggle)
    button_confirm.on_clicked(on_confirm)
    
    # Initial display update
    update_display()
    
    plt.tight_layout()
    plt.show()
    
    # Return the final coordinates if confirmed, otherwise return None
    if params['confirmed']:
        return calculate_source_coords()
    else:
        print("Light source coordinate setup cancelled.")
        return None

def crop_and_interpolate_images(image_stack, target_size, left=None, top=None, right=None, bottom=None, rotation_degrees=0, rotate_90_left=False):
    """
    Crop and interpolate an image stack to a target size using explicit crop boundaries with optional rotation.
    
    Args:
        image_stack: Tensor of shape [num_images, 3, height, width] for RGB images
        target_size: Desired output size (height, width) or single int for square output
        left: Left crop boundary (x-coordinate)
        top: Top crop boundary (y-coordinate)
        right: Right crop boundary (x-coordinate)
        bottom: Bottom crop boundary (y-coordinate)
        rotation_degrees: Rotation angle in degrees to apply around crop center before cropping
        rotate_90_left: If True, rotate the final cropped images 90 degrees counterclockwise
    
    Returns:
        Processed tensor of shape [num_images, 3, target_height, target_width]
    """
    num_images, channels, height, width = image_stack.shape
    
    # Handle single int target size
    if isinstance(target_size, int):
        target_height = target_width = target_size
    else:
        target_height, target_width = target_size
    
    # Calculate crop center
    crop_center_x = (left + right) / 2
    crop_center_y = (top + bottom) / 2
    crop_width = right - left
    crop_height = bottom - top
    
    # Print crop dimensions for debugging
    print(f"Cropping region: ({left}, {top}) to ({right}, {bottom}), size: {crop_width}x{crop_height}")
    if rotation_degrees != 0:
        print(f"Applying rotation: {rotation_degrees} degrees around center ({crop_center_x}, {crop_center_y})")
    
    # Create output tensor
    processed_stack = torch.zeros(num_images, channels, target_height, target_width)
    
    # Process each image
    for i in range(num_images):
        current_image = image_stack[i]  # Shape: [3, H, W]
        
        # Apply rotation if specified
        if rotation_degrees != 0:
            # Convert rotation to radians
            angle_rad = math.radians(rotation_degrees)
            
            # Create rotation matrix
            cos_theta = math.cos(angle_rad)
            sin_theta = math.sin(angle_rad)
            
            # Create affine transformation matrix for rotation around crop center
            # First translate to center, then rotate, then translate back
            # PyTorch expects the inverse transformation matrix
            
            # Normalize coordinates to [-1, 1] range for grid_sample
            norm_center_x = 2.0 * crop_center_x / width - 1.0
            norm_center_y = 2.0 * crop_center_y / height - 1.0
            
            # Create affine transformation matrix (inverse transformation)
            # For rotation around a point: T^-1 * R^-1 * T
            affine_matrix = torch.tensor([
                [cos_theta, sin_theta, -norm_center_x * cos_theta - norm_center_y * sin_theta + norm_center_x],
                [-sin_theta, cos_theta, norm_center_x * sin_theta - norm_center_y * cos_theta + norm_center_y]
            ], dtype=torch.float32)
            
            # Create grid for sampling
            grid = F.affine_grid(
                affine_matrix.unsqueeze(0),  # Add batch dimension
                [1, channels, height, width],
                align_corners=False
            )
            
            # Apply rotation using grid sampling
            rotated_image = F.grid_sample(
                current_image.unsqueeze(0),  # Add batch dimension
                grid,
                mode='bilinear',
                padding_mode='reflection',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
            
            current_image = rotated_image
        
        # Crop the image (after rotation if applied)
        cropped = current_image[:, top:bottom, left:right]
        
        # Resize to target dimensions using bicubic interpolation
        resized = F.interpolate(
            cropped.unsqueeze(0),  # Add batch dimension
            size=(target_height, target_width),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        # Apply 90-degree left rotation if requested
        if rotate_90_left:
            # Rotate 90 degrees counterclockwise: transpose and flip vertically
            # For tensor [C, H, W], we transpose dims 1,2 then flip dim 1
            resized = torch.transpose(resized, 1, 2)  # [C, W, H]
            resized = torch.flip(resized, [1])        # [C, W, H] flipped vertically
        
        # Store in output tensor
        processed_stack[i] = resized
    
    return processed_stack


# Main script execution
if __name__ == "__main__":
    
    # Load configuration from JSON file
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Extract configuration values
    num_layers = config['training']['num_layers']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    scatter = config['training']['scatter']
    use_lr_scheduler = config['training'].get('use_lr_scheduler', False)  # Default to False
    
    image_size = config['image_processing']['image_size']
    input_pad = config['image_processing']['input_pad']
    disp_pos = config['image_processing']['disp_pos']
    num_channels = config['image_processing']['num_channels']
    
    img_x_um = config['physical_parameters']['img_x_um']
    sample_z_um = config['physical_parameters']['sample_z_um']
    
    z_depth_mm = config['light_source']['z_depth_mm']
    right_to_left = config['light_source']['right_to_left']
    light_step = config['light_source']['light_step']
    center_image = config['light_source']['center_image']
    y_position = config['light_source']['y_position']
    intensity_mode = config['light_source']['intensity_mode']
    NA = config['light_source']['NA']
    blur_sigma = config['light_source']['blur_sigma']
    
    folder_path = config['data_paths']['folder_path']
    output_base_path = config['data_paths']['output_base_path']
    
    left = config['image_cropping']['left']
    top = config['image_cropping']['top']
    right = config['image_cropping']['right']
    bottom = config['image_cropping']['bottom']
    target_size = tuple(config['image_cropping']['target_size'])
    rotation_degrees = config['image_cropping']['rotation_degrees']
    rotate_90_left = config['image_cropping']['rotate_90_left']
    
    use_gui_default = config['interface']['use_gui_default']

    D_layer = np.round(sample_z_um / img_x_um * image_size / num_layers).astype(int)

    print(f"Configuration loaded:")
    print(f"  - Training: {num_layers} layers, {num_epochs} epochs")
    print(f"  - Physical: FOV={img_x_um}μm, Sample={sample_z_um}μm")
    print(f"  - Light: depth={z_depth_mm}mm, step={light_step}mm")
    print(f"  - Data: {folder_path}")
    
    # Load RGB images (3 channels)
    image_stack = load_images(folder_path)
    print(f'image stack shape: {image_stack.shape}')

    # Apply cropping and interpolation with parameters from config
    image_stack = crop_and_interpolate_images(
        image_stack,
        target_size=target_size,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        rotation_degrees=rotation_degrees,
        rotate_90_left=rotate_90_left
    )
    print(f'Processed image stack shape: {image_stack.shape}')

    # Create base output directory first
    os.makedirs(output_base_path, exist_ok=True)

    # save cropped image for verification
    plt.figure(figsize=(8, 6))
    plt.imshow(image_stack[disp_pos].permute(1, 2, 0).numpy())
    plt.title('Cropped Image')
    plt.axis('off')
    plt.savefig(f'{output_base_path}/cropped_image_verification.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # save original image for comparison
    plt.figure(figsize=(8, 6))
    original_image = Image.open(os.path.join(folder_path, os.listdir(folder_path)[disp_pos]))
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig(f'{output_base_path}/original_image_reference.png', dpi=150, bbox_inches='tight')
    plt.close()

    image_stack = image_stack.to(device)
    # clip between 0 and 1
    image_stack = torch.clamp(image_stack, 0, 1)
    
    num_images = image_stack.shape[0]
    print(f"Loaded {num_images} RGB images with shape {image_stack.shape[1:]} each.")
    print(f'image stack max: {image_stack.max()}, min: {image_stack.min()}')
    
    # Use GUI setting from configuration
    use_gui = use_gui_default
    print(f"Using GUI for light source coordinates: {use_gui}")
    
    if use_gui:
        print("Opening interactive GUI for light source coordinate setup...")
        # Move image_stack back to CPU for GUI display
        image_stack_cpu = image_stack.cpu()
        source_coords = interactive_source_coords_gui(image_stack_cpu, image_size, num_images, config)
        
        if source_coords is None:
            print("GUI cancelled, falling back to automatic coordinate generation.")
            source_coords = gen_source_coords(num_images, image_size, img_x_um, z_depth_mm, right_to_left, light_step, center_image, y_position)
        else:
            print("Using coordinates from interactive GUI.")
    else:
        print("Using automatic coordinate generation.")
        source_coords = gen_source_coords(num_images, image_size, img_x_um, z_depth_mm, right_to_left, light_step, center_image, y_position)
    
    print(f"Generated {len(source_coords)} source coordinates.")
    
    # Generate synthetic incident light distributions (now in RGB)
    incident = []
    for pos in range(num_images):
        source_coord = source_coords[pos]
        synthetic = generate_synthetic_incident_light(
            image_size=image_size,
            source_coord=source_coord,
            margin=input_pad,
            NA=NA,
            intensity_mode=intensity_mode,
            blur_sigma=blur_sigma,
            device=device,
            num_channels=num_channels)
        incident.append(synthetic)
    
    print(f'synthetic incident shape is {incident[0].shape}')
    incident_size = incident[0].shape[1]  # Get height/width from dimension 1 since dim 0 is now channels
    
    # Initialize alpha parameter for RGB (add a dimension for color channels)
    alpha_init_raw = image_stack[num_images//2].type(torch.float32)  # shape (3, height, width)
    
    # compute padding needed
    diff = incident_size - alpha_init_raw.shape[1]  # Shape: [3, H, W], so look at dim 1
    pad_each_side = diff // 2  
    padding = (pad_each_side, pad_each_side, pad_each_side, pad_each_side)  # (left, right, top, bottom)
    
    # Handle each RGB channel
    alpha_channels = []
    for c in range(3):  # RGB channels
        channel = alpha_init_raw[c].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        padded_channel = F.pad(channel, padding, mode='reflect')  # [1, 1, incident_size, incident_size]
        padded_channel = padded_channel.squeeze(0).squeeze(0)  # [incident_size, incident_size]
        alpha_channels.append(padded_channel)
        
    # Stack channels [incident_size, incident_size, 3]
    alpha_stacked = torch.stack(alpha_channels, dim=-1)
    
    # Add layer dimension and repeat
    # [incident_size, incident_size, num_layers, 3]
    alpha_padded = alpha_stacked.unsqueeze(2).repeat(1, 1, num_layers, 1)
    
    # Scale alpha values 
    for c in range(3):
        alpha_padded[..., c] = alpha_padded[..., c] ** (1/num_layers)
        
    # Create parameter
    alpha_padded = alpha_padded.to(device)
    alpha_torch = torch.nn.Parameter(alpha_padded)
    
    # Set up optimizer
    optimizer = Adam([alpha_torch], lr=learning_rate)
    
    # Conditionally set up learning rate scheduler
    if use_lr_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: linear_lr(epoch, num_epochs))
        print("Learning rate scheduler enabled")
    else:
        scheduler = None
        print("Learning rate scheduler disabled")
    
    torch.autograd.set_detect_anomaly(True)
    
    # Training loop
    loss_values = []
    for epoch in range(num_epochs):
        for pos in range(num_images):
            optimizer.zero_grad()
            predicted, layer_outputs = forward_model_end(incident[pos], alpha_torch, scatter, source_coords[pos], D_layer)
            loss = loss_function_alpha(predicted, layer_outputs, image_stack[pos], incident[pos], alpha_torch)
            loss.backward()
            optimizer.step()
            
        # Step the learning rate scheduler if enabled
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = learning_rate
            
        loss_values.append(loss.item())
        print(f"Epoch {epoch}, LR = {current_lr:.6f}, Loss: {loss.item()}")
    
    # Create output directories
    os.makedirs(f'{output_base_path}/predicted', exist_ok=True)
    os.makedirs(f'{output_base_path}/incident', exist_ok=True)
    os.makedirs(f'{output_base_path}/layers', exist_ok=True)
    os.makedirs(f'{output_base_path}/alpha_layers', exist_ok=True)
    os.makedirs(f'{output_base_path}/alpha_inv_layers', exist_ok=True)
    os.makedirs(f'{output_base_path}/ground_truth', exist_ok=True)
    os.makedirs(f'{output_base_path}/error_maps', exist_ok=True)
    os.makedirs(f'{output_base_path}/resolution_analysis', exist_ok=True)

    # Save a copy of this script and config
    import shutil
    shutil.copyfile(__file__, f"{output_base_path}/forward_model.py")
    
    # Save the configuration used for documentation
    if len(sys.argv) > 1:
        config_filename = os.path.basename(sys.argv[1])
        shutil.copyfile(sys.argv[1], f"{output_base_path}/{config_filename}")
    else:
        # If no config file specified, save current parameters as JSON for documentation
        import json
        current_config = {
            "training": {
                "num_layers": num_layers,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "scatter": scatter,
                "use_lr_scheduler": use_lr_scheduler
            },
            "image_processing": {
                "image_size": image_size,
                "input_pad": input_pad,
                "disp_pos": disp_pos
            },
            "note": "This config was auto-generated from hardcoded values"
        }
        with open(f"{output_base_path}/config_used.json", "w") as f:
            json.dump(current_config, f, indent=2)
    
    # save loss log
    with open(f"{output_base_path}/loss_log.txt", "w") as f:
        for val in loss_values:
            f.write(str(val) + "\n")
            
    # Save the learned alpha layers
    with torch.no_grad():
        # Convert alpha_torch to numpy for visualization
        alpha_np = alpha_torch.detach().cpu().numpy()
        
        # Save each alpha layer as an image
        for layer_idx in range(num_layers):
            # For RGB, we save a color image
            alpha_layer = alpha_np[..., layer_idx, :]  # Shape: [H, W, 3]
            
            # Crop to match final image size if needed
            if input_pad > 0:
                # Crop each channel separately
                alpha_r = crop_incident_to_final(alpha_layer[..., 0], input_pad)
                alpha_g = crop_incident_to_final(alpha_layer[..., 1], input_pad) 
                alpha_b = crop_incident_to_final(alpha_layer[..., 2], input_pad)
                
                # Recombine channels for color image
                alpha_rgb = np.stack([alpha_r, alpha_g, alpha_b], axis=2)
                alpha_inv_rgb = 1 - alpha_rgb
            else:
                alpha_rgb = alpha_layer
                alpha_inv_rgb = 1 - alpha_layer
                
            # Save as PNG (RGB)
            alpha_rgb = np.clip(alpha_rgb, 0, 1)  # Ensure values are in [0, 1]
            alpha_inv_rgb = np.clip(alpha_inv_rgb, 0, 1)  # Ensure values are in [0, 1]
            plt.imsave(f'{output_base_path}/alpha_layers/alpha_layer_{layer_idx}.png', 
                       alpha_rgb)
            plt.imsave(f'{output_base_path}/alpha_inv_layers/alpha_inv_layer_{layer_idx}.png', 
                       alpha_inv_rgb)
    
    # Perform Fourier resolution analysis on alpha layers
    if FOURIER_ANALYSIS_AVAILABLE:
        print("\nPerforming Fourier resolution analysis on alpha layers...")
        try:
            # Calculate physical parameters for resolution analysis
            pixels_per_micron_xy = image_size / img_x_um  # XY spatial resolution
            microns_per_layer = sample_z_um / num_layers  # Z layer spacing
            
            print(f"Physical parameters:")
            print(f"  XY resolution: {pixels_per_micron_xy:.2f} pixels/µm")
            print(f"  Layer spacing: {microns_per_layer:.2f} µm")
            print(f"  Total Z depth: {sample_z_um:.1f} µm")
            
            # Run Fourier analysis on alpha_torch
            resolution_results = analyze_alpha_layers_z_resolution(
                alpha_torch=alpha_torch,
                pixels_per_micron_xy=pixels_per_micron_xy,
                microns_per_layer=microns_per_layer,
                output_path=f'{output_base_path}/resolution_analysis',
                show_plot=True,
                manual_threshold=0.1  # Use consistent threshold
            )
            
            print("Fourier resolution analysis completed successfully!")
            print(f"Results saved to: {output_base_path}/resolution_analysis/")
            print("Generated files:")
            print("  - alpha_resolution_analysis.png (overview)")
            print("  - alpha_resolution_results.txt (detailed results)")
            print("  - layer_fourier_analysis/ (individual layer graphics)")
            print("  - alpha_layers_grid_*.png (layer overview grids)")
            
        except Exception as e:
            print(f"Error during Fourier resolution analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping Fourier resolution analysis (module not available)")
    
    # Extract all output images and calculate error maps
    layer_np_imgs = []
    all_error_maps = []  # Store all error maps for averaging
    error_statistics = []  # Store statistics for each error map
    
    for pos in range(num_images):
        with torch.no_grad():
            # Get the predictions and layer outputs
            pred, layer_outputs = forward_model_end(incident[pos], alpha_torch, scatter, source_coords[pos], D_layer)
            
            # Convert to numpy for visualization
            final_pred_np = pred.detach().cpu().numpy()  # Shape: [3, H, W] for RGB
            final_pred_np = crop_incident_to_final(final_pred_np, input_pad)
            
            # For saving RGB images, rearrange to [H, W, 3]
            final_pred_rgb = np.transpose(final_pred_np, (1, 2, 0))
            incident_np = incident[pos].cpu().numpy()  # Shape: [3, H, W]
            incident_np = crop_incident_to_final(incident_np, input_pad)  # Crop first
            incident_rgb = np.transpose(incident_np, (1, 2, 0))  # Then transpose to [H, W, 3]
            #print(f'final_pred_rgb shape: {final_pred_rgb.shape}, incident_rgb shape: {incident_rgb.shape}')
            
            # Save predicted RGB image and incident light
            #print(f'final_pred_rgb shape: {final_pred_rgb.shape}')
            plt.imsave(f'{output_base_path}/predicted/pos_{pos}.png', 
                       final_pred_rgb)
            plt.imsave(f'{output_base_path}/incident/pos_{pos}.png', 
                       incident_rgb)
            
            # Save ground truth (original input images - already at correct size)
            ground_truth_np = image_stack[pos].cpu().numpy()  # Shape: [3, H, W] - already correct size
            ground_truth_rgb = np.transpose(ground_truth_np, (1, 2, 0))  # Convert to [H, W, 3]
            plt.imsave(f'{output_base_path}/ground_truth/pos_{pos}.png', ground_truth_rgb)
            
            # Debug: check shapes before error calculation
            print(f"Position {pos}: pred shape {final_pred_np.shape}, gt shape {ground_truth_np.shape}")
            
            # Generate and save error map for this position
            # prediction is cropped from padded size, ground truth is already correct size
            error_map, stats = save_error_map(
                final_pred_np,  # [3, H, W] - cropped from padded prediction
                ground_truth_np,  # [3, H, W] - original size, no cropping needed
                f'{output_base_path}/error_maps/error_map_pos_{pos}.png',
                title=f'Prediction Error - Position {pos}',
                colormap='hot'  # Hot colormap shows errors well (black=low, red/yellow=high)
            )
            
            all_error_maps.append(error_map)
            error_statistics.append(stats)
            print(f"Error map {pos}: Mean={stats['mean']:.4f}, Max={stats['max']:.4f}, Std={stats['std']:.4f}")
    
            # Process layer outputs
            layer_np = []
            for j in range(num_layers):
                layer_output = layer_outputs[j].detach().cpu().numpy()  # [3, H, W]
                cropped_output = crop_incident_to_final(layer_output, input_pad)
                layer_np.append(cropped_output)
    
                # Save layer output as RGB image
                layer_rgb = np.transpose(cropped_output, (1, 2, 0))
                plt.imsave(f'{output_base_path}/layers/layer_{j}_pos_{pos}.png', 
                           layer_rgb)
                
            layer_np_imgs.append(layer_np)
    
    # Calculate and save average error map
    if all_error_maps:
        avg_error_map = np.mean(all_error_maps, axis=0)
        
        # Calculate statistics for average error map
        avg_stats = {
            'mean': np.mean(avg_error_map),
            'max': np.max(avg_error_map), 
            'std': np.std(avg_error_map)
        }
        
        # Save average error map with proper normalization
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(avg_error_map, cmap='hot', aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Average Absolute Error', rotation=270, labelpad=20, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Set title and labels
        ax.set_title('Average Prediction Error Across All Positions', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        
        # Add statistics text
        stats_text = f'Mean Error: {avg_stats["mean"]:.4f}\nMax Error: {avg_stats["max"]:.4f}\nStd Error: {avg_stats["std"]:.4f}\nNumber of Images: {len(all_error_maps)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{output_base_path}/error_maps/average_error_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nAverage error map: Mean={avg_stats['mean']:.4f}, Max={avg_stats['max']:.4f}, Std={avg_stats['std']:.4f}")
        
        # Save error statistics to file
        with open(f'{output_base_path}/error_maps/error_statistics.txt', 'w') as f:
            f.write("Error Map Statistics\n")
            f.write("===================\n\n")
            
            f.write("Individual Position Statistics:\n")
            for pos, stats in enumerate(error_statistics):
                f.write(f"Position {pos}: Mean={stats['mean']:.6f}, Max={stats['max']:.6f}, Std={stats['std']:.6f}\n")
            
            f.write(f"\nAverage Across All Positions:\n")
            f.write(f"Mean Error: {avg_stats['mean']:.6f}\n")
            f.write(f"Max Error: {avg_stats['max']:.6f}\n") 
            f.write(f"Std Error: {avg_stats['std']:.6f}\n")
            f.write(f"Number of Images: {len(all_error_maps)}\n")
            
            # Calculate overall statistics
            all_means = [s['mean'] for s in error_statistics]
            all_maxs = [s['max'] for s in error_statistics]
            all_stds = [s['std'] for s in error_statistics]
            
            f.write(f"\nStatistics Across Positions:\n")
            f.write(f"Mean of means: {np.mean(all_means):.6f} ± {np.std(all_means):.6f}\n")
            f.write(f"Mean of maxs: {np.mean(all_maxs):.6f} ± {np.std(all_maxs):.6f}\n")
            f.write(f"Mean of stds: {np.mean(all_stds):.6f} ± {np.std(all_stds):.6f}\n")
    
    # Now visualize the light propagation through layers for a sample position
    layer_np_imgs = np.array(layer_np_imgs)  # Shape: [num_positions, num_layers, 3, height, width]
    print(f'layer_np_imgs shape(disp_pos, layer number, channels, x, y) : {layer_np_imgs.shape}')
    
    # For RGB visualization, we'll create a grid of images
    plt.figure(figsize=(15, 5))
    
    # Show input light
    plt.subplot(1, num_layers+2, 1)
    incident_img = np.transpose(crop_incident_to_final(incident[disp_pos].cpu().numpy(), input_pad), (1, 2, 0))
    plt.imshow(incident_img)
    plt.title('Simulated Incident Light')
    plt.axis('off')
    
    # Show each layer
    for i in range(num_layers):
        plt.subplot(1, num_layers+2, i+2)
        layer_img = np.transpose(layer_np_imgs[disp_pos, i], (1, 2, 0))
        plt.imshow(layer_img)
        plt.title(f'Layer {i}')
        plt.axis('off')
    
    # Show ground truth
    plt.subplot(1, num_layers+2, num_layers+2)
    gt_img = np.transpose(image_stack[disp_pos].cpu().numpy(), (1, 2, 0))
    plt.imshow(gt_img)
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_base_path}/layer_visualization.png', dpi=300)
    plt.close()  # Close the figure to free memory