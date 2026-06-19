import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
import glob
import torch
from collections import defaultdict

def get_wavelength_from_channel(channel):
    """
    Convert channel number to wavelength.
    Channel 0 = 703nm, Channel 29 = 452nm (linear interpolation)
    """
    return 703 - (channel * (703 - 452) / 29)

def analyze_alpha_layers_z_resolution(alpha_torch, pixels_per_micron_xy, microns_per_layer, 
                                    output_path=None, show_plot=True, manual_threshold=None):
    """
    Analyze Z-axis resolution of alpha layers using Fourier analysis.
    
    Args:
        alpha_torch: Alpha tensor of shape [H, W, num_layers, 3] or [H, W, num_layers]
        pixels_per_micron_xy: Spatial resolution in XY plane (pixels per micron)
        microns_per_layer: Physical distance between layers in microns
        output_path: Path to save results (optional)
        show_plot: Whether to show visualization plots
        manual_threshold: Manual threshold for resolution analysis (0.0-1.0)
        
    Returns:
        dict: Dictionary containing resolution analysis results for each channel
    """
    
    # Convert to numpy if it's a torch tensor
    if hasattr(alpha_torch, 'cpu'):
        alpha_np = alpha_torch.detach().cpu().numpy()
    else:
        alpha_np = alpha_torch
    
    print(f"Analyzing alpha layers with shape: {alpha_np.shape}")
    
    # Handle different input shapes
    if alpha_np.ndim == 3:  # [H, W, num_layers] - single channel
        alpha_np = alpha_np[..., np.newaxis]  # Add channel dimension -> [H, W, num_layers, 1]
        num_channels = 1
        channel_names = ['Grayscale']
    elif alpha_np.ndim == 4:  # [H, W, num_layers, 3] - RGB
        num_channels = alpha_np.shape[3]
        channel_names = ['Red', 'Green', 'Blue'] if num_channels == 3 else [f'Channel_{i}' for i in range(num_channels)]
    else:
        raise ValueError(f"Expected 3D or 4D alpha array, got shape {alpha_np.shape}")
    
    H, W, num_layers, _ = alpha_np.shape
    
    # Calculate layer resolution (layers per micron in Z)
    layers_per_micron_z = 1.0 / microns_per_layer
    
    results = {}
    
    # Process each color channel
    for ch in range(num_channels):
        channel_name = channel_names[ch]
        print(f"\nAnalyzing {channel_name} channel...")
        
        # Extract 2D slices along different axes for this channel
        alpha_channel = alpha_np[:, :, :, ch]  # [H, W, num_layers]
        
        # Analyze resolution along Z-axis by taking slices through the volume
        z_resolutions = []
        xy_resolutions = []
        
        # Method 1: Analyze XY resolution for each Z layer
        xy_res_per_layer = []
        for layer_idx in range(num_layers):
            layer_slice = alpha_channel[:, :, layer_idx]  # [H, W]
            
            try:
                resolution_xy, cutoff_xy = analyze_image_resolution_from_array(
                    layer_slice, pixels_per_micron_xy, pixels_per_micron_xy,
                    manual_threshold=manual_threshold, show_plot=False
                )
                xy_res_per_layer.append(resolution_xy)
            except:
                xy_res_per_layer.append(np.inf)
        
        # Method 2: Analyze Z resolution by taking XY slices through the volume
        z_res_profiles = []
        
        # Sample multiple lines through the volume for Z analysis
        sample_points = [(H//4, W//4), (H//2, W//2), (3*H//4, 3*W//4), 
                        (H//4, 3*W//4), (3*H//4, W//4)]
        
        for y, x in sample_points:
            z_profile = alpha_channel[y, x, :]  # [num_layers] - 1D profile along Z
            
            try:
                resolution_z, cutoff_z = analyze_1d_resolution(
                    z_profile, layers_per_micron_z, manual_threshold=manual_threshold
                )
                z_res_profiles.append(resolution_z)
            except:
                z_res_profiles.append(np.inf)
        
        # Calculate statistics
        valid_xy = [r for r in xy_res_per_layer if r != np.inf]
        valid_z = [r for r in z_res_profiles if r != np.inf]
        
        channel_results = {
            'channel_name': channel_name,
            'xy_resolution_per_layer': xy_res_per_layer,
            'z_resolution_profiles': z_res_profiles,
            'mean_xy_resolution': np.mean(valid_xy) if valid_xy else np.inf,
            'std_xy_resolution': np.std(valid_xy) if valid_xy else 0,
            'mean_z_resolution': np.mean(valid_z) if valid_z else np.inf,
            'std_z_resolution': np.std(valid_z) if valid_z else 0,
            'valid_xy_layers': len(valid_xy),
            'valid_z_profiles': len(valid_z)
        }
        
        results[channel_name] = channel_results
        
        print(f"  XY resolution: {channel_results['mean_xy_resolution']:.2f} ± {channel_results['std_xy_resolution']:.2f} µm ({channel_results['valid_xy_layers']}/{num_layers} layers)")
        print(f"  Z resolution: {channel_results['mean_z_resolution']:.2f} ± {channel_results['std_z_resolution']:.2f} µm ({channel_results['valid_z_profiles']}/{len(sample_points)} profiles)")
    
    # Create visualization
    if show_plot:
        plot_alpha_resolution_analysis(alpha_np, results, microns_per_layer, output_path)
    
    # Generate detailed graphics for each individual layer
    if output_path:
        generate_individual_alpha_layer_graphics(
            alpha_np, pixels_per_micron_xy, microns_per_layer, 
            output_path, manual_threshold
        )
    
    # Save results
    if output_path:
        save_alpha_resolution_results(results, alpha_np.shape, microns_per_layer, output_path)
    
    return results

def analyze_image_resolution_from_array(image_array, pixels_per_micron_x, pixels_per_micron_y, 
                                      manual_threshold=None, show_plot=False):
    """
    Analyze resolution of a 2D numpy array (modified from original function).
    """
    # Ensure image is 2D
    if len(image_array.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {image_array.shape}")
    
    image = image_array.astype(np.float32)
    
    # Skip normalization if image is already in a reasonable range
    if np.max(image) <= 1.0 and np.min(image) >= 0.0:
        pass  # Already normalized
    else:
        # normalize to 0-1
        image_range = np.max(image) - np.min(image)
        if image_range > 0:
            image = (image - np.min(image)) / image_range
        else:
            return np.inf, 0  # No variation in image
    
    # Get actual image dimensions
    rows, cols = image.shape
    
    # --- 1. Fourier Transform ---
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    power_spectrum = np.abs(f_transform_shifted)**2
    
    # --- 2. Threshold Detection ---
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    radius = np.sqrt(x*x + y*y)
    
    # Calculate radial profile by binning
    max_radius = int(np.min([crow, ccol]))
    radial_bins = np.arange(0, max_radius, 1)
    radial_profile = np.zeros(len(radial_bins))
    
    for i, r in enumerate(radial_bins):
        mask = (radius >= r) & (radius < r + 1)
        if np.any(mask):
            radial_profile[i] = np.mean(power_spectrum[mask])
    
    # Apply threshold detection
    log_profile = np.log1p(radial_profile + 1e-10)  # Add small epsilon to avoid log(0)
    
    if manual_threshold is not None:
        threshold_value = manual_threshold * 255
    else:
        # Use Otsu's method on the log of the radial profile
        normalized = ((log_profile - log_profile.min()) / (log_profile.max() - log_profile.min()) * 255).astype(np.uint8)
        threshold_value, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to original scale
    threshold_power = np.exp((threshold_value / 255.0) * (log_profile.max() - log_profile.min()) + log_profile.min()) - 1
    cutoff_idx = np.where(radial_profile <= threshold_power)[0]
    cutoff_frequency_pixels = radial_bins[cutoff_idx[0]] if len(cutoff_idx) > 0 else max_radius
    
    # Calculate resolution in microns
    resolution_microns_x = 1 / (cutoff_frequency_pixels / cols) * (1 / pixels_per_micron_x) if cutoff_frequency_pixels != 0 else float('inf')
    resolution_microns_y = 1 / (cutoff_frequency_pixels / rows) * (1 / pixels_per_micron_y) if cutoff_frequency_pixels != 0 else float('inf')
    resolution_microns = (resolution_microns_x + resolution_microns_y) / 2
    
    # Convert cutoff frequency to cycles/um
    avg_image_size = (rows + cols) / 2
    normalized_frequency = cutoff_frequency_pixels / (avg_image_size / 2)
    avg_pixels_per_micron = (pixels_per_micron_x + pixels_per_micron_y) / 2
    nyquist_frequency = avg_pixels_per_micron / 2
    cutoff_frequency_cycles_per_um = normalized_frequency * nyquist_frequency
    
    return resolution_microns, cutoff_frequency_cycles_per_um

def analyze_1d_resolution(profile_1d, samples_per_micron, manual_threshold=None):
    """
    Analyze resolution of a 1D profile (for Z-axis analysis).
    """
    if len(profile_1d) < 4:  # Too few points for meaningful analysis
        return np.inf, 0
    
    # Normalize profile
    profile = profile_1d.astype(np.float32)
    profile_range = np.max(profile) - np.min(profile)
    if profile_range > 0:
        profile = (profile - np.min(profile)) / profile_range
    else:
        return np.inf, 0  # No variation
    
    # 1D FFT
    f_transform = np.fft.fft(profile)
    f_transform_shifted = np.fft.fftshift(f_transform)
    power_spectrum = np.abs(f_transform_shifted)**2
    
    # Create frequency bins
    N = len(profile)
    freqs = np.arange(N) - N//2
    
    # Calculate power profile (already 1D)
    log_profile = np.log1p(power_spectrum + 1e-10)
    
    if manual_threshold is not None:
        threshold_value = manual_threshold * 255
    else:
        # Use Otsu's method
        normalized = ((log_profile - log_profile.min()) / (log_profile.max() - log_profile.min()) * 255).astype(np.uint8)
        threshold_value, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to original scale
    threshold_power = np.exp((threshold_value / 255.0) * (log_profile.max() - log_profile.min()) + log_profile.min()) - 1
    
    # Find cutoff frequency (from center outward)
    center = N // 2
    cutoff_idx = center
    
    for i in range(1, center):
        if (power_spectrum[center + i] <= threshold_power and 
            power_spectrum[center - i] <= threshold_power):
            cutoff_idx = i
            break
    
    # Calculate resolution
    if cutoff_idx > 0:
        cutoff_frequency_normalized = cutoff_idx / (N / 2)
        nyquist_frequency = samples_per_micron / 2
        cutoff_frequency_per_micron = cutoff_frequency_normalized * nyquist_frequency
        resolution_microns = 1.0 / cutoff_frequency_per_micron if cutoff_frequency_per_micron > 0 else np.inf
    else:
        resolution_microns = np.inf
        cutoff_frequency_per_micron = 0
    
    return resolution_microns, cutoff_frequency_per_micron

def plot_alpha_resolution_analysis(alpha_np, results, microns_per_layer, output_path=None):
    """
    Create visualization plots for alpha layer resolution analysis.
    """
    H, W, num_layers, num_channels = alpha_np.shape
    
    fig, axes = plt.subplots(2, num_channels + 1, figsize=(15, 8))
    if num_channels == 1:
        axes = axes.reshape(2, -1)
    
    # Plot alpha volume slices for each channel
    for ch in range(num_channels):
        channel_name = list(results.keys())[ch]
        alpha_channel = alpha_np[:, :, :, ch]
        
        # Top row: XY slice (middle layer)
        middle_layer = num_layers // 2
        im1 = axes[0, ch].imshow(alpha_channel[:, :, middle_layer], cmap='viridis')
        axes[0, ch].set_title(f'{channel_name} - Layer {middle_layer}')
        axes[0, ch].set_xlabel('X (pixels)')
        axes[0, ch].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[0, ch])
        
        # Bottom row: XZ slice (middle Y)
        middle_y = H // 2
        xz_slice = alpha_channel[middle_y, :, :]  # [W, num_layers]
        im2 = axes[1, ch].imshow(xz_slice.T, cmap='viridis', aspect='auto')
        axes[1, ch].set_title(f'{channel_name} - XZ Slice (Y={middle_y})')
        axes[1, ch].set_xlabel('X (pixels)')
        axes[1, ch].set_ylabel('Z Layer')
        plt.colorbar(im2, ax=axes[1, ch])
    
    # Plot resolution statistics
    ax_stats = axes[:, -1]
    
    # Top subplot: XY resolution per layer
    layers = np.arange(num_layers)
    for ch, (channel_name, result) in enumerate(results.items()):
        xy_res = result['xy_resolution_per_layer']
        valid_res = [r if r != np.inf else np.nan for r in xy_res]
        ax_stats[0].plot(layers, valid_res, 'o-', label=channel_name, alpha=0.7)
    
    ax_stats[0].set_xlabel('Layer Index')
    ax_stats[0].set_ylabel('XY Resolution (µm)')
    ax_stats[0].set_title('XY Resolution per Layer')
    ax_stats[0].legend()
    ax_stats[0].grid(True, alpha=0.3)
    
    # Bottom subplot: Summary statistics
    ax_stats[1].axis('off')
    
    stats_text = "Resolution Analysis Summary:\n\n"
    for channel_name, result in results.items():
        stats_text += f"{channel_name}:\n"
        stats_text += f"  XY: {result['mean_xy_resolution']:.2f} ± {result['std_xy_resolution']:.2f} µm\n"
        stats_text += f"  Z:  {result['mean_z_resolution']:.2f} ± {result['std_z_resolution']:.2f} µm\n"
        stats_text += f"  Valid XY: {result['valid_xy_layers']}/{num_layers}\n"
        stats_text += f"  Valid Z: {result['valid_z_profiles']}/5\n\n"
    
    stats_text += f"Layer spacing: {microns_per_layer:.2f} µm\n"
    stats_text += f"Total depth: {num_layers * microns_per_layer:.1f} µm"
    
    ax_stats[1].text(0.05, 0.95, stats_text, transform=ax_stats[1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(f'{output_path}/alpha_resolution_analysis.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_individual_alpha_layer_graphics(alpha_torch, pixels_per_micron_xy, microns_per_layer, 
                                           output_path, manual_threshold=0.1):
    """
    Generate detailed Fourier analysis graphics for each individual alpha layer.
    Works with both RGB and grayscale images.
    """
    # Convert to numpy if needed
    if hasattr(alpha_torch, 'cpu'):
        alpha_np = alpha_torch.detach().cpu().numpy()
    else:
        alpha_np = alpha_torch
    
    # Handle different input shapes
    if alpha_np.ndim == 3:  # [H, W, num_layers] - single channel
        alpha_np = alpha_np[..., np.newaxis]  # Add channel dimension -> [H, W, num_layers, 1]
        num_channels = 1
        channel_names = ['Grayscale']
    elif alpha_np.ndim == 4:  # [H, W, num_layers, 3] - RGB
        num_channels = alpha_np.shape[3]
        channel_names = ['Red', 'Green', 'Blue'] if num_channels == 3 else [f'Channel_{i}' for i in range(num_channels)]
    else:
        raise ValueError(f"Expected 3D or 4D alpha array, got shape {alpha_np.shape}")
    
    H, W, num_layers, _ = alpha_np.shape
    
    # Create output directory for individual layer analysis
    layer_graphics_path = os.path.join(output_path, 'layer_fourier_analysis')
    os.makedirs(layer_graphics_path, exist_ok=True)
    
    print(f"Generating Fourier analysis graphics for {num_layers} layers...")
    
    # Process each layer
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers-1}...")
        
        # For RGB: create a subplot for each channel, for grayscale: single plot
        if num_channels == 1:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, axes = plt.subplots(num_channels, 3, figsize=(18, 6 * num_channels))
            if num_channels == 1:
                axes = axes.reshape(1, -1)
        
        layer_results = []
        
        for ch in range(num_channels):
            channel_name = channel_names[ch]
            layer_slice = alpha_np[:, :, layer_idx, ch]  # [H, W]
            
            # Skip if layer is empty or has no variation
            if np.std(layer_slice) < 1e-10:
                print(f"  Skipping {channel_name} channel - no variation")
                continue
            
            try:
                # Analyze this layer
                resolution, cutoff_freq = analyze_layer_with_visualization(
                    layer_slice, pixels_per_micron_xy, pixels_per_micron_xy,
                    manual_threshold, axes[ch] if num_channels > 1 else axes,
                    f'{channel_name} - Layer {layer_idx}', layer_idx, microns_per_layer
                )
                
                layer_results.append({
                    'channel': channel_name,
                    'resolution': resolution,
                    'cutoff_freq': cutoff_freq
                })
                
            except Exception as e:
                print(f"  Error analyzing {channel_name} channel: {e}")
                continue
        
        # Add overall title and save
        fig.suptitle(f'Fourier Analysis - Alpha Layer {layer_idx} (Depth: {layer_idx * microns_per_layer:.1f} μm)', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
        
        # Save the figure
        output_file = os.path.join(layer_graphics_path, f'layer_{layer_idx:02d}_fourier_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save layer results to text file
        results_file = os.path.join(layer_graphics_path, f'layer_{layer_idx:02d}_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Fourier Analysis Results - Layer {layer_idx}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Depth: {layer_idx * microns_per_layer:.2f} μm\n")
            f.write(f"Physical resolution: {pixels_per_micron_xy:.2f} pixels/μm\n\n")
            
            for result in layer_results:
                f.write(f"{result['channel']} Channel:\n")
                f.write(f"  Resolution: {result['resolution']:.3f} μm\n")
                f.write(f"  Cutoff frequency: {result['cutoff_freq']:.3f} cycles/μm\n\n")
    
    print(f"Individual layer analysis complete! Graphics saved to: {layer_graphics_path}")
    
    # Also create a summary grid showing all layers
    create_layer_summary_grid(alpha_np, pixels_per_micron_xy, microns_per_layer, output_path, channel_names)

def create_layer_summary_grid(alpha_np, pixels_per_micron_xy, microns_per_layer, output_path, channel_names):
    """
    Create a summary grid showing all alpha layers at once for quick overview.
    """
    H, W, num_layers, num_channels = alpha_np.shape
    
    # Determine grid size (try to make it roughly square)
    grid_cols = int(np.ceil(np.sqrt(num_layers)))
    grid_rows = int(np.ceil(num_layers / grid_cols))
    
    for ch in range(num_channels):
        channel_name = channel_names[ch]
        
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3*grid_cols, 3*grid_rows))
        if grid_rows == 1:
            axes = axes.reshape(1, -1)
        elif grid_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for layer_idx in range(num_layers):
            row = layer_idx // grid_cols
            col = layer_idx % grid_cols
            ax = axes[row, col]
            
            layer_slice = alpha_np[:, :, layer_idx, ch]
            
            # Display the layer
            im = ax.imshow(layer_slice, cmap='viridis')
            ax.set_title(f'Layer {layer_idx}\n{layer_idx * microns_per_layer:.1f} μm', fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for layer_idx in range(num_layers, grid_rows * grid_cols):
            row = layer_idx // grid_cols
            col = layer_idx % grid_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Alpha Layers Overview - {channel_name} Channel', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the grid
        grid_file = os.path.join(output_path, f'alpha_layers_grid_{channel_name.lower()}.png')
        plt.savefig(grid_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Layer grid saved: {grid_file}")

def analyze_layer_with_visualization(image_array, pixels_per_micron_x, pixels_per_micron_y, 
                                   manual_threshold, axes, title, layer_idx, microns_per_layer):
    """
    Analyze a single layer and create visualization plots similar to the original function.
    """
    # Ensure image is 2D
    if len(image_array.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {image_array.shape}")
    
    image = image_array.astype(np.float32)
    
    # Normalize to 0-1
    if np.max(image) <= 1.0 and np.min(image) >= 0.0:
        pass  # Already normalized
    else:
        image_range = np.max(image) - np.min(image)
        if image_range > 0:
            image = (image - np.min(image)) / image_range
        else:
            return np.inf, 0  # No variation in image
    
    # Get actual image dimensions
    rows, cols = image.shape
    
    # --- 1. Fourier Transform ---
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    power_spectrum = np.abs(f_transform_shifted)**2
    
    # --- 2. Threshold Detection ---
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    radius = np.sqrt(x*x + y*y)
    
    # Calculate radial profile by binning
    max_radius = int(np.min([crow, ccol]))
    radial_bins = np.arange(0, max_radius, 1)
    radial_profile = np.zeros(len(radial_bins))
    
    for i, r in enumerate(radial_bins):
        mask = (radius >= r) & (radius < r + 1)
        if np.any(mask):
            radial_profile[i] = np.mean(power_spectrum[mask])
    
    # Apply threshold detection
    log_profile = np.log1p(radial_profile + 1e-10)
    
    if manual_threshold is not None:
        threshold_value = manual_threshold * 255
        threshold_method = f"Manual ({manual_threshold})"
    else:
        normalized = ((log_profile - log_profile.min()) / (log_profile.max() - log_profile.min()) * 255).astype(np.uint8)
        threshold_value, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_method = "Otsu"
    
    # Convert back to original scale
    threshold_power = np.exp((threshold_value / 255.0) * (log_profile.max() - log_profile.min()) + log_profile.min()) - 1
    cutoff_idx = np.where(radial_profile <= threshold_power)[0]
    cutoff_frequency_pixels = radial_bins[cutoff_idx[0]] if len(cutoff_idx) > 0 else max_radius
    
    # Calculate resolution in microns
    resolution_microns_x = 1 / (cutoff_frequency_pixels / cols) * (1 / pixels_per_micron_x) if cutoff_frequency_pixels != 0 else float('inf')
    resolution_microns_y = 1 / (cutoff_frequency_pixels / rows) * (1 / pixels_per_micron_y) if cutoff_frequency_pixels != 0 else float('inf')
    resolution_microns = (resolution_microns_x + resolution_microns_y) / 2
    
    # Convert cutoff frequency to cycles/um
    avg_image_size = (rows + cols) / 2
    normalized_frequency = cutoff_frequency_pixels / (avg_image_size / 2)
    avg_pixels_per_micron = (pixels_per_micron_x + pixels_per_micron_y) / 2
    nyquist_frequency = avg_pixels_per_micron / 2
    cutoff_frequency_cycles_per_um = normalized_frequency * nyquist_frequency
    
    # --- 3. Visualization ---
    # Original Image
    axes[0].imshow(image, cmap='viridis')
    axes[0].set_title(f'{title}\nResolution: {resolution_microns:.2f} μm')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    
    # Power Spectrum with Resolution Circle
    axes[1].imshow(np.log1p(power_spectrum), cmap='hot')
    circle = plt.Circle((ccol, crow), cutoff_frequency_pixels, color='cyan', fill=False, linewidth=2)
    axes[1].add_patch(circle)
    axes[1].set_title(f'Power Spectrum ({threshold_method})')
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')
    
    # Radial Profile in cycles/um
    radial_bins_cycles_per_um = []
    for r in radial_bins:
        norm_freq = r / (avg_image_size / 2)
        cycles_per_um = norm_freq * nyquist_frequency
        radial_bins_cycles_per_um.append(cycles_per_um)
    
    axes[2].plot(radial_bins_cycles_per_um, radial_profile, 'b-', linewidth=2, label='Radial Profile')
    axes[2].axvline(x=cutoff_frequency_cycles_per_um, color='cyan', linestyle='--', 
                   linewidth=2, label=f'Cutoff: {cutoff_frequency_cycles_per_um:.2f} cyc/μm')
    axes[2].axhline(y=threshold_power, color='red', linestyle=':', linewidth=1, 
                   alpha=0.7, label=f'{threshold_method} threshold')
    axes[2].set_xlabel('Spatial Frequency (cycles/μm)')
    axes[2].set_ylabel('Average Power')
    axes[2].set_title('Radial Power Profile')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Add depth information
    depth_text = f'Depth: {layer_idx * microns_per_layer:.1f} μm'
    axes[2].text(0.02, 0.98, depth_text, transform=axes[2].transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return resolution_microns, cutoff_frequency_cycles_per_um

def save_alpha_resolution_results(results, alpha_shape, microns_per_layer, output_path):
    """
    Save resolution analysis results to file.
    """
    os.makedirs(output_path, exist_ok=True)
    
    with open(f'{output_path}/alpha_resolution_results.txt', 'w') as f:
        f.write("Alpha Layers Resolution Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Alpha tensor shape: {alpha_shape}\n")
        f.write(f"Layer spacing: {microns_per_layer:.3f} µm\n")
        f.write(f"Total depth: {alpha_shape[2] * microns_per_layer:.1f} µm\n\n")
        
        for channel_name, result in results.items():
            f.write(f"{channel_name} Channel Results:\n")
            f.write("-" * 30 + "\n")
            f.write(f"XY Resolution: {result['mean_xy_resolution']:.3f} ± {result['std_xy_resolution']:.3f} µm\n")
            f.write(f"Z Resolution:  {result['mean_z_resolution']:.3f} ± {result['std_z_resolution']:.3f} µm\n")
            f.write(f"Valid XY layers: {result['valid_xy_layers']}/{alpha_shape[2]}\n")
            f.write(f"Valid Z profiles: {result['valid_z_profiles']}/5\n\n")
            
            # Detailed per-layer XY resolution
            f.write("Per-layer XY resolution (µm):\n")
            for i, res in enumerate(result['xy_resolution_per_layer']):
                if res != np.inf:
                    f.write(f"  Layer {i:2d}: {res:.3f}\n")
                else:
                    f.write(f"  Layer {i:2d}: Invalid\n")
            f.write("\n")
    
    print(f"Resolution analysis results saved to: {output_path}/alpha_resolution_results.txt")

def analyze_image_resolution(image_path, pixels_per_micron_x, pixels_per_micron_y, channel=None, manual_threshold=None, show_plot=False):
    """
    Analyzes the resolution of a biological image using its Fourier transform and Otsu's method.

    Args:
        image_path (str): The path to the image file.
        pixels_per_micron_x (float): The number of pixels per micron in the x-direction.
        pixels_per_micron_y (float): The number of pixels per micron in the y-direction.
        channel (int, optional): Channel number for multi-channel images (e.g., TIF stacks).
        manual_threshold (float, optional): Manual threshold value (0.0-1.0). If None, uses Otsu's method.
        show_plot (bool): Whether to display the visualization plots.

    Returns:
        tuple: A tuple containing the estimated resolution in microns and the cutoff frequency in cycles/um.
    """
    # Load the image
    if image_path.lower().endswith('.tif') or image_path.lower().endswith('.tiff'):
        # Load TIFF with tifffile
        image_data = tifffile.imread(image_path)
        if channel is not None:
            if len(image_data.shape) == 3:
                image = image_data[channel]
            else:
                raise ValueError(f"Channel {channel} specified but image is not multi-channel")
        else:
            if len(image_data.shape) == 3:
                image = image_data[0]  # Use first channel if none specified
            else:
                image = image_data
    else:
        # Load regular image formats in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

    # Ensure image is 2D
    if len(image.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Get actual image dimensions
    rows, cols = image.shape
    if show_plot:
        print(f"Loaded image dimensions: {cols}px x {rows}px")

    # normalize to 0-1
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # --- 1. Fourier Transform ---
    # Perform 2D FFT
    f_transform = np.fft.fft2(image)
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform)
    # Calculate the power spectrum (magnitude squared)
    power_spectrum = np.abs(f_transform_shifted)**2

    # --- 2. Threshold Detection ---
    # Create a grid of radial distances from the center
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    radius = np.sqrt(x*x + y*y)

    # Calculate radial profile by binning
    max_radius = int(np.min([crow, ccol]))
    radial_bins = np.arange(0, max_radius, 1)
    radial_profile = np.zeros(len(radial_bins))
    
    for i, r in enumerate(radial_bins):
        mask = (radius >= r) & (radius < r + 1)
        if np.any(mask):
            radial_profile[i] = np.mean(power_spectrum[mask])
    
    # Apply threshold detection
    log_profile = np.log1p(radial_profile)
    
    if manual_threshold is not None:
        # Use manual threshold
        threshold_value = manual_threshold * 255
        threshold_method = f"Manual ({manual_threshold})"
    else:
        # Use Otsu's method on the log of the radial profile
        # Normalize to 0-255 for Otsu
        normalized = ((log_profile - log_profile.min()) / (log_profile.max() - log_profile.min()) * 255).astype(np.uint8)
        threshold_value, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_method = "Otsu"
    
    # Convert back to original scale
    threshold_power = np.exp((threshold_value / 255.0) * (log_profile.max() - log_profile.min()) + log_profile.min()) - 1
    cutoff_idx = np.where(radial_profile <= threshold_power)[0]
    cutoff_frequency_pixels = radial_bins[cutoff_idx[0]] if len(cutoff_idx) > 0 else max_radius

    # Calculate resolution in microns
    # Resolution is the inverse of the cutoff frequency
    resolution_microns_x = 1 / (cutoff_frequency_pixels / cols) * (1 / pixels_per_micron_x) if cutoff_frequency_pixels != 0 else float('inf')
    resolution_microns_y = 1 / (cutoff_frequency_pixels / rows) * (1 / pixels_per_micron_y) if cutoff_frequency_pixels != 0 else float('inf')
    # Average the resolution in x and y directions
    resolution_microns = (resolution_microns_x + resolution_microns_y) / 2

    # Convert cutoff frequency to cycles/um
    # Normalized frequency = cutoff_frequency_pixels / (image_size/2)
    # Cycles/um = normalized_frequency * Nyquist_frequency
    # Nyquist frequency = pixels_per_micron / 2
    avg_image_size = (rows + cols) / 2
    normalized_frequency = cutoff_frequency_pixels / (avg_image_size / 2)
    avg_pixels_per_micron = (pixels_per_micron_x + pixels_per_micron_y) / 2
    nyquist_frequency = avg_pixels_per_micron / 2
    cutoff_frequency_cycles_per_um = normalized_frequency * nyquist_frequency

    # --- 3. Visualization ---
    if show_plot:
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        if channel is not None:
            plt.title(f'Est. Resolution: {resolution_microns:.2f} µm, (ch {channel})')
        else:
            plt.title(f'Est. Resolution: {resolution_microns:.2f} µm')
        plt.axis('off')

        # Power Spectrum with Resolution Threshold
        plt.subplot(1, 3, 2)
        plt.imshow(np.log1p(power_spectrum), cmap='hot')
        circle = plt.Circle((ccol, crow), cutoff_frequency_pixels, color='cyan', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.title(f'Power Spectrum ({threshold_method})')
        plt.axis('off')

        # Radial Profile in cycles/um
        plt.subplot(1, 3, 3)
        # Convert radial_bins to cycles/um
        radial_bins_cycles_per_um = []
        for r in radial_bins:
            norm_freq = r / (avg_image_size / 2)
            cycles_per_um = norm_freq * nyquist_frequency
            radial_bins_cycles_per_um.append(cycles_per_um)
        
        plt.plot(radial_bins_cycles_per_um, radial_profile, 'b-', linewidth=2)
        plt.axvline(x=cutoff_frequency_cycles_per_um, color='cyan', linestyle='--', linewidth=2)
        plt.axhline(y=threshold_power, color='red', linestyle=':', linewidth=1, alpha=0.7, label=f'{threshold_method} threshold')
        plt.xlabel('Spatial Frequency (cycles/µm)')
        plt.ylabel('Average Power')
        plt.title('Radial Power Profile')
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return resolution_microns, cutoff_frequency_cycles_per_um

def process_batch_images(base_path, pixels_per_micron, display_channel=20, manual_threshold=0.1):
    """
    Process all images in the directory and calculate statistics.
    For hyperspectral images, process all 30 channels and average results.
    """
    # Dictionary to store results by image type
    results = defaultdict(list)
    
    # Find all image files
    image_patterns = ['mini_raw_*.tif', 'hs_raw_*.tif', 'hs_gen_*.tif']
    
    for pattern in image_patterns:
        image_type = pattern.split('_')[0] + '_' + pattern.split('_')[1]  # e.g., 'mini_raw', 'hs_raw', 'hs_gen'
        files = glob.glob(os.path.join(base_path, pattern))
        files.sort()  # Sort to ensure consistent ordering
        
        print(f"\nProcessing {image_type} images:")
        print("-" * 40)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            image_number = filename.split('_')[-1].split('.')[0]  # Extract number from filename
            
            try:
                # Show plot only for image 6 and display_channel
                show_plot = (image_number == '6')
                
                if image_type == 'mini_raw':
                    # mini_raw images are grayscale, process single channel
                    resolution, cutoff_freq = analyze_image_resolution(
                        file_path,
                        pixels_per_micron,
                        pixels_per_micron,
                        channel=None,
                        manual_threshold=manual_threshold,
                        show_plot=show_plot
                    )
                    
                    results[image_type].append({
                        'filename': filename,
                        'resolution': resolution,
                        'cutoff_frequency': cutoff_freq
                    })
                    
                    print(f"{filename}: Resolution = {resolution:.2f} µm, Cutoff = {cutoff_freq:.2f} cycles/µm")
                    
                else:
                    # hs_raw and hs_gen: process all 30 channels
                    if show_plot:
                        print(f"\nProcessing all 30 channels for {filename} (displaying channel {display_channel})")
                    
                    channel_resolutions = []
                    channel_cutoffs = []
                    
                    for channel in range(30):  # Process all 30 channels
                        # Only show plot for display_channel
                        show_channel_plot = show_plot and (channel == display_channel)
                        
                        try:
                            resolution, cutoff_freq = analyze_image_resolution(
                                file_path,
                                pixels_per_micron,
                                pixels_per_micron,
                                channel=channel,
                                manual_threshold=manual_threshold,
                                show_plot=show_channel_plot
                            )
                            
                            if resolution != float('inf'):
                                channel_resolutions.append(resolution)
                                channel_cutoffs.append(cutoff_freq)
                                
                        except Exception as e:
                            print(f"  Error processing channel {channel}: {e}")
                    
                    # Calculate average across all valid channels
                    if channel_resolutions:
                        avg_resolution = np.mean(channel_resolutions)
                        avg_cutoff = np.mean(channel_cutoffs)
                        std_resolution = np.std(channel_resolutions)
                        std_cutoff = np.std(channel_cutoffs)
                        
                        results[image_type].append({
                            'filename': filename,
                            'resolution': avg_resolution,
                            'cutoff_frequency': avg_cutoff,
                            'resolution_std': std_resolution,
                            'cutoff_std': std_cutoff,
                            'valid_channels': len(channel_resolutions)
                        })
                        
                        print(f"{filename}: Resolution = {avg_resolution:.2f} ± {std_resolution:.2f} µm, "
                              f"Cutoff = {avg_cutoff:.2f} ± {std_cutoff:.2f} cycles/µm "
                              f"(averaged over {len(channel_resolutions)} channels)")
                    else:
                        print(f"{filename}: No valid channels processed")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results

def process_batch_images_with_wavelength(base_path, pixels_per_micron, display_channel=20, manual_threshold=0.1):
    """
    Process all images in the directory and calculate statistics.
    For hyperspectral images, process all 30 channels and store results per channel.
    """
    # Dictionary to store results by image type and channel
    results = defaultdict(lambda: defaultdict(list))
    
    # Find all image files
    image_patterns = ['mini_raw_*.tif', 'hs_raw_*.tif', 'hs_gen_*.tif']
    
    for pattern in image_patterns:
        image_type = pattern.split('_')[0] + '_' + pattern.split('_')[1]  # e.g., 'mini_raw', 'hs_raw', 'hs_gen'
        files = glob.glob(os.path.join(base_path, pattern))
        files.sort()  # Sort to ensure consistent ordering
        
        print(f"\nProcessing {image_type} images:")
        print("-" * 40)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            image_number = filename.split('_')[-1].split('.')[0]  # Extract number from filename
            
            try:
                # Show plot only for image 6 and display_channel
                show_plot = (image_number == '6')
                
                if image_type == 'mini_raw':
                    # mini_raw images are grayscale, process single channel
                    resolution, cutoff_freq = analyze_image_resolution(
                        file_path,
                        pixels_per_micron,
                        pixels_per_micron,
                        channel=None,
                        manual_threshold=manual_threshold,
                        show_plot=show_plot
                    )
                    
                    results[image_type]['single'].append({
                        'filename': filename,
                        'resolution': resolution,
                        'cutoff_frequency': cutoff_freq
                    })
                    
                    print(f"{filename}: Resolution = {resolution:.2f} µm, Cutoff = {cutoff_freq:.2f} cycles/µm")
                    
                else:
                    # hs_raw and hs_gen: process all 30 channels
                    if show_plot:
                        print(f"\nProcessing all 30 channels for {filename} (displaying channel {display_channel})")
                    
                    for channel in range(30):  # Process all 30 channels
                        # Only show plot for display_channel
                        show_channel_plot = show_plot and (channel == display_channel)
                        
                        try:
                            resolution, cutoff_freq = analyze_image_resolution(
                                file_path,
                                pixels_per_micron,
                                pixels_per_micron,
                                channel=channel,
                                manual_threshold=manual_threshold,
                                show_plot=show_channel_plot
                            )
                            
                            if resolution != float('inf'):
                                wavelength = get_wavelength_from_channel(channel)
                                results[image_type][channel].append({
                                    'filename': filename,
                                    'wavelength': wavelength,
                                    'resolution': resolution,
                                    'cutoff_frequency': cutoff_freq
                                })
                                
                        except Exception as e:
                            print(f"  Error processing channel {channel}: {e}")
                    
                    # Print summary for this image
                    channel_resolutions = []
                    for channel in range(30):
                        if results[image_type][channel]:
                            channel_data = [d for d in results[image_type][channel] if d['filename'] == filename]
                            if channel_data:
                                channel_resolutions.append(channel_data[0]['resolution'])
                    
                    if channel_resolutions:
                        avg_resolution = np.mean(channel_resolutions)
                        std_resolution = np.std(channel_resolutions)
                        print(f"{filename}: Average resolution = {avg_resolution:.2f} ± {std_resolution:.2f} µm "
                              f"(over {len(channel_resolutions)} channels)")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results

def plot_resolution_vs_wavelength(results):
    """
    Plot resolution as a function of wavelength for hyperspectral images.
    """
    plt.figure(figsize=(6, 4))  # Smaller figure for journal
    
    colors = {'hs_raw': '#1f77b4', 'hs_gen': '#d62728'}  # Blue and red
    labels = {'hs_raw': 'Ground Truth', 'hs_gen': 'Generated'}
    
    for image_type in ['hs_raw', 'hs_gen']:
        if image_type not in results:
            continue
            
        wavelengths = []
        mean_resolutions = []
        
        # Process each channel
        for channel in range(30):
            if channel in results[image_type] and results[image_type][channel]:
                channel_data = results[image_type][channel]
                resolutions = [d['resolution'] for d in channel_data if d['resolution'] != float('inf')]
                
                if resolutions:
                    wavelength = get_wavelength_from_channel(channel)
                    wavelengths.append(wavelength)
                    mean_resolutions.append(np.mean(resolutions))
        
        if wavelengths:
            wavelengths = np.array(wavelengths)
            mean_resolutions = np.array(mean_resolutions)
            
            # Sort by wavelength (ascending order: 450nm to 700nm)
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            mean_resolutions = mean_resolutions[sort_idx]
            
            # Plot without error bars
            plt.plot(wavelengths, mean_resolutions, 
                    color=colors[image_type], label=labels[image_type], 
                    marker='o', markersize=6, linewidth=2.5)
            
            print(f"\n{labels[image_type]} wavelength analysis:")
            for wl, res in zip(wavelengths, mean_resolutions):
                print(f"  {wl:.0f} nm: {res:.2f} µm")
    
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('Resolution (µm)', fontsize=14)
    plt.title('Spatial Resolution vs Wavelength', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(440, 710)
    plt.ylim(5, 16)
    
    # Increase tick font size
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return wavelengths, mean_resolutions

def calculate_statistics(results):
    """
    Calculate and display statistics for each image type.
    """
    print("\n" + "="*60)
    print("RESOLUTION ANALYSIS SUMMARY")
    print("="*60)
    
    for image_type in results.keys():
        if image_type == 'mini_raw':
            # Handle single channel images
            if 'single' in results[image_type]:
                data = results[image_type]['single']
                resolutions = [d['resolution'] for d in data if d['resolution'] != float('inf')]
                cutoff_freqs = [d['cutoff_frequency'] for d in data if d['resolution'] != float('inf')]
                
                if resolutions:
                    mean_res = np.mean(resolutions)
                    std_res = np.std(resolutions)
                    mean_cutoff = np.mean(cutoff_freqs)
                    std_cutoff = np.std(cutoff_freqs)
                    
                    print(f"\n{image_type.upper()} Images (n={len(resolutions)}):")
                    print(f"  Resolution: {mean_res:.2f} ± {std_res:.2f} µm")
                    print(f"  Cutoff Frequency: {mean_cutoff:.2f} ± {std_cutoff:.2f} cycles/µm")
        else:
            # Handle hyperspectral images - calculate overall statistics
            all_resolutions = []
            all_cutoffs = []
            
            for channel in range(30):
                if channel in results[image_type]:
                    channel_data = results[image_type][channel]
                    for d in channel_data:
                        if d['resolution'] != float('inf'):
                            all_resolutions.append(d['resolution'])
                            all_cutoffs.append(d['cutoff_frequency'])
            
            if all_resolutions:
                mean_res = np.mean(all_resolutions)
                std_res = np.std(all_resolutions)
                mean_cutoff = np.mean(all_cutoffs)
                std_cutoff = np.std(all_cutoffs)
                
                print(f"\n{image_type.upper()} Images (all channels, n={len(all_resolutions)}):")
                print(f"  Resolution: {mean_res:.2f} ± {std_res:.2f} µm")
                print(f"  Cutoff Frequency: {mean_cutoff:.2f} ± {std_cutoff:.2f} cycles/µm")

def test_alpha_resolution_analysis():
    """
    Test function to verify the alpha layer resolution analysis works.
    """
    print("Testing alpha layer resolution analysis...")
    
    # Create synthetic alpha tensor for testing
    H, W, num_layers = 128, 128, 16
    
    # Create a synthetic alpha tensor with some structure
    alpha_test = np.zeros((H, W, num_layers, 3))
    
    # Add some vertical structures (should show up in XY analysis)
    for layer in range(num_layers):
        # Add vertical stripes with varying frequency
        x = np.linspace(0, 10*np.pi, W)
        y = np.linspace(0, 10*np.pi, H)
        X, Y = np.meshgrid(x, y)
        
        # Create patterns that change with layer (Z structure)
        frequency_factor = 1 + layer * 0.1
        pattern = 0.5 + 0.3 * np.sin(frequency_factor * X) * np.cos(frequency_factor * Y)
        
        # Add some depth-dependent changes
        depth_factor = np.exp(-0.1 * layer)  # Exponential decay with depth
        pattern *= depth_factor
        
        # Apply to all RGB channels
        for c in range(3):
            alpha_test[:, :, layer, c] = pattern
    
    # Test the analysis
    results = analyze_alpha_layers_z_resolution(
        alpha_torch=alpha_test,
        pixels_per_micron_xy=2.0,  # 2 pixels per micron
        microns_per_layer=1.0,     # 1 micron per layer
        output_path='./test_output',
        show_plot=True,
        manual_threshold=0.1
    )
    
    print("Test completed successfully!")
    return results

if __name__ == '__main__':
    # Check if we're testing the alpha analysis or running the original analysis
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_alpha_resolution_analysis()
    else:
        # Original analysis code
        # Configuration
        BASE_PATH = '/media/al/Extreme SSD/20250701_usaf/deconvolution_results/batch_reg/img_6'
        DISPLAY_CHANNEL = 23  # Channel to display in visualizations
        IMAGE_WIDTH_MICRONS = 940
        MANUAL_THRESHOLD = 0.1  # Set to None to use Otsu's method

        # Calculate pixels per micron (assuming square images)
        # We'll use a sample image to get dimensions
        sample_files = glob.glob(os.path.join(BASE_PATH, '*.tif'))
        if sample_files:
            sample_image = tifffile.imread(sample_files[0])
            if len(sample_image.shape) == 3:
                IMAGE_WIDTH_PIXELS = sample_image.shape[2]
            else:
                IMAGE_WIDTH_PIXELS = sample_image.shape[1]
            
            PIXELS_PER_MICRON = IMAGE_WIDTH_PIXELS / IMAGE_WIDTH_MICRONS
            print(f"Sample image dimensions: {IMAGE_WIDTH_PIXELS} pixels")
            print(f"Physical width: {IMAGE_WIDTH_MICRONS} µm")
            print(f"Pixels per micron: {PIXELS_PER_MICRON:.2f}")
            
            # Print wavelength mapping
            print(f"\nWavelength mapping:")
            for ch in [0, 5, 10, 15, 20, 25, 29]:
                wl = get_wavelength_from_channel(ch)
                print(f"  Channel {ch}: {wl:.0f} nm")
        else:
            raise FileNotFoundError(f"No TIF files found in {BASE_PATH}")

        # Process all images with wavelength analysis
        results = process_batch_images_with_wavelength(
            BASE_PATH,
            PIXELS_PER_MICRON,
            display_channel=DISPLAY_CHANNEL,
            manual_threshold=MANUAL_THRESHOLD
        )
        
        # Calculate and display statistics
        calculate_statistics(results)
        
        # Plot resolution vs wavelength
        print("\nGenerating resolution vs wavelength plot...")
        plot_resolution_vs_wavelength(results)