import numpy as np
import imageio.v3 as imageio
import os
import pyvista as pv
from scipy.ndimage import zoom, gaussian_filter
import cv2
import glob
import re


def load_sequential_images(directory, base_filename="alpha_layer_", extension=".png", rotate_90_right=True):
    """
    Load sequentially numbered images and return a normalized 3D volume.
    Optionally rotate each image 90 degrees to the right (clockwise) after loading.

    Parameters:
    - directory: Folder containing the images
    - base_filename: Common filename prefix (e.g., "alpha_layer_")
    - extension: File extension (e.g., ".png")
    - rotate_90_right: If True, rotate each image 90° clockwise after load

    Returns:
    - volume: 3D numpy array (z, y, x) normalized to [0, 1]
    """
    directory = os.path.join(directory, 'alpha_layers')

    # Get all matching files
    all_files = [f for f in os.listdir(directory)
                if f.startswith(base_filename) and f.endswith(extension)]
    # Extract numbers from filenames
    file_numbers = []
    for filename in all_files:
        number_part = filename[len(base_filename):-len(extension)]
        if number_part.isdigit():
            file_numbers.append((int(number_part), filename))
    # Sort by number
    file_numbers.sort()
    sorted_files = [f[1] for f in file_numbers]
    if not sorted_files:
        raise ValueError(f"No images found matching pattern {base_filename}N{extension}")
    print(f"Loading {len(sorted_files)} sequential images")
    # Load all images
    images = [imageio.imread(os.path.join(directory, f)) for f in sorted_files]
    # Optionally rotate images 90 degrees clockwise
    if rotate_90_right:
        print("Applying 90° clockwise rotation to all images")
        images = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in images]
    # Handle different image formats (RGB vs grayscale)
    processed_images = []
    for img in images:
        if len(img.shape) == 3:  # RGB or RGBA image
            processed_images.append(img[:, :, 0])  # Taking first channel
        else:  # Grayscale image
            processed_images.append(img)
    # Stack images into volume
    volume = np.stack(processed_images, axis=0)
    # Normalize volume
    volume = volume.astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    return volume

def apply_3d_gaussian_blur(volume, sigma):
    """
    Apply a 3D Gaussian blur to the volume data.
    Parameters:
    - volume: 3D numpy array
    - sigma: Standard deviation for Gaussian kernel (can be a single value or a sequence for each axis)
    Returns:
    - Blurred volume
    """
    return gaussian_filter(volume, sigma=sigma)

def normalize_with_gamma(volume, gamma=2.2, min_clip=None, max_clip=None):
    # Make a copy to avoid modifying the original
    processed = volume.astype(np.float32)
    # Optional clipping to remove outliers
    if min_clip is not None or max_clip is not None:
        if min_clip is not None:
            if 0 <= min_clip <= 1:
                min_val = np.percentile(processed, min_clip * 100)
            else:
                min_val = min_clip
            processed = np.maximum(processed, min_val)
        if max_clip is not None:
            if 0 <= max_clip <= 1:
                max_val = np.percentile(processed, max_clip * 100)
            else:
                max_val = max_clip
            processed = np.minimum(processed, max_val)
    # Normalize to [0, 1] range
    min_val = np.min(processed)
    max_val = np.max(processed)
    # Avoid division by zero
    if max_val > min_val:
        processed = (processed - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(processed)
    # Apply gamma correction
    processed = np.power(processed, 1.0 / gamma)
    return processed

def create_gamma_corrected_colorbar(cmap="plasma", gamma=2.2, orientation="vertical", 
                                  width=100, height=400, output_path="colorbar.png"):
    """
    Create and save a gamma-corrected colorbar image.
    
    Parameters:
    - cmap: Colormap name (e.g., "plasma", "viridis")
    - gamma: Gamma correction value
    - orientation: "vertical" or "horizontal"
    - width: Width of the colorbar in pixels
    - height: Height of the colorbar in pixels
    - output_path: Path to save the colorbar image
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import PowerNorm
    
    # Create figure and axis
    if orientation == "vertical":
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        # Create vertical gradient
        gradient = np.linspace(0, 1, height).reshape(height, 1)
    else:  # horizontal
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        # Create horizontal gradient
        gradient = np.linspace(0, 1, width).reshape(1, width)
    
    # Apply gamma correction to the gradient
    gamma_corrected_gradient = np.power(gradient, 1.0 / gamma)
    
    # Display the gradient with the specified colormap
    im = ax.imshow(gamma_corrected_gradient, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    
    # Remove axes and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Remove any padding/margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save the colorbar
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    print(f"Colorbar saved to: {output_path}")

def create_colorbars_with_labels(cmap="plasma", gamma=2.2, output_dir="colorbars", 
                               colorbar_width=80, colorbar_height=400):
    """
    Create vertical and horizontal colorbars with labels and tick marks.
    Apply gamma correction the same way as normalize_with_gamma function.
    
    Parameters:
    - cmap: Colormap name
    - gamma: Gamma correction value
    - output_dir: Directory to save the colorbars
    - colorbar_width: Width of vertical colorbar
    - colorbar_height: Height of vertical colorbar
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a linear gradient from 0 to 1
    linear_values = np.linspace(0, 1, 256)
    
    # Apply gamma correction exactly like normalize_with_gamma function
    gamma_corrected_values = np.power(linear_values, 1.0 / gamma)
    
    # Get the colormap
    colormap = plt.cm.get_cmap(cmap)
    
    # Create colors using the gamma-corrected values
    colors = colormap(gamma_corrected_values)
    
    # Create a custom colormap from the gamma-corrected colors
    gamma_corrected_cmap = mcolors.ListedColormap(colors)
    
    # Vertical colorbar
    fig, ax = plt.subplots(figsize=(2, 6))
    
    # Create a mappable object for the colorbar using linear normalization
    # but with our gamma-corrected colormap
    sm = plt.cm.ScalarMappable(cmap=gamma_corrected_cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    # Create vertical colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Absorbance', rotation=270, labelpad=20, fontsize=12)
    
    # Set only 0 and 1 tick locations to avoid overlap
    tick_locations = [0.0, 1.0]
    cbar.set_ticks(tick_locations)
    cbar.set_ticklabels(['0', '1'])
    cbar.ax.tick_params(labelsize=12)
    
    # Remove the main axes
    ax.remove()
    
    # Save vertical colorbar
    vert_path = os.path.join(output_dir, f"vertical_colorbar_{cmap}_gamma{gamma:.2f}.png")
    plt.savefig(vert_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    
    # Horizontal colorbar
    fig, ax = plt.subplots(figsize=(6, 1.5))
    
    # Create a mappable object for the colorbar using linear normalization
    # but with our gamma-corrected colormap
    sm = plt.cm.ScalarMappable(cmap=gamma_corrected_cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    # Create horizontal colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.8)
    cbar.set_label('Absorbance', fontsize=12)
    
    # Set only 0 and 1 tick locations to avoid overlap
    cbar.set_ticks(tick_locations)
    cbar.set_ticklabels(['0', '1'])
    cbar.ax.tick_params(labelsize=12)
    
    # Remove the main axes
    ax.remove()
    
    # Save horizontal colorbar
    horiz_path = os.path.join(output_dir, f"horizontal_colorbar_{cmap}_gamma{gamma:.2f}.png")
    plt.savefig(horiz_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    
    print(f"Vertical colorbar saved to: {vert_path}")
    print(f"Horizontal colorbar saved to: {horiz_path}")

def visualize_volume(volume, zoom_factor_z=1, title="Volume Visualization",
                  cmap="viridis", background_color="black"):
    # Normalize volume
    volume = volume.astype(np.float32)
    volume /= volume.max()
    # Interpolate along z-axis if needed
    if zoom_factor_z != 1:
        volume = zoom(volume, (zoom_factor_z, 1, 1), order=1)
    print(f"Volume shape: {volume.shape}")
    # Create ImageData object
    nz, ny, nx = volume.shape
    grid = pv.ImageData(dimensions=(nz, nx, ny))
    # Set spacing and origin
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    # Add volume data
    grid.point_data["values"] = volume.ravel(order='F')
    # Create plotter with theme
    plotter = pv.Plotter(theme=pv.themes.DarkTheme())
    # Set background color
    plotter.set_background(background_color)
    # Add volume
    plotter.add_volume(grid,
                      cmap=cmap,
                      diffuse=0.7,
                      clim=[0, 1],
                      show_scalar_bar=False)
    plotter.add_text(title, position=(10, 10), color='white')
    plotter.show()
def create_rotating_volume_frames(volume, output_path, n_frames=180, zoom_factor_z=1.5,
                                title="Volume Visualization", cmap="viridis",
                                background_color="black", resolution=(1920, 1080)):
    """
    Create and save a rotating volume visualization as individual image frames.
    Parameters:
    - volume: 3D numpy array containing volume data
    - output_path: Base path to save the frames (a folder will be created)
    - n_frames: Number of frames for the full 360° rotation
    - zoom_factor_z: Factor to zoom the z dimension
    - title: Title text for the visualization
    - cmap: Colormap for the volume
    - background_color: Background color of the scene
    - resolution: Output image resolution as (width, height)
    """
    # Normalize volume
    volume = volume.astype(np.float32)
    volume /= volume.max()
    # Interpolate along z-axis if needed
    if zoom_factor_z != 1:
        volume = zoom(volume, (zoom_factor_z, 1, 1), order=1)
    print(f"Volume shape: {volume.shape}")
    # Create ImageData object
    nz, ny, nx = volume.shape
    grid = pv.ImageData(dimensions=(nz, nx, ny))
    # Set spacing and origin
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    # Add volume data
    grid.point_data["values"] = volume.ravel(order='F')
    # Create off-screen plotter
    plotter = pv.Plotter(off_screen=True, theme=pv.themes.DarkTheme())
    plotter.window_size = resolution
    plotter.set_background(background_color)
    # Add volume using same settings as visualize_volume
    plotter.add_volume(grid, cmap=cmap, diffuse=0.7, clim=[0, 1], show_scalar_bar=False)
    plotter.add_text(title, position=(10, 10), color='white', font_size=20)
    # Set explicit camera position (matches visualize_volume)
    # This creates a view similar to the interactive display
    focal_point = grid.center
    position = [nz*2, ny/2, nx/2]  # Position the camera far enough back
    plotter.camera_position = [position, focal_point, [0, 1, 0]]
    # Create a path for frames
    frame_dir = os.path.splitext(output_path)[0] + '/frames'
    os.makedirs(frame_dir, exist_ok=True)
    # Generate frames with slower rotation for stability
    angles = np.linspace(0, 360, n_frames, endpoint=False)
    print(f"Rendering {n_frames} frames...")
    for i, angle in enumerate(angles):
        print(f"Rendering frame {i+1}/{n_frames}...", end='\r')
        # Reset to initial position, then rotate by angle
        plotter.reset_camera()
        plotter.view_isometric()
        # Apply rotation manually - set absolute azimuth instead of calling it
        plotter.camera.azimuth = angle
        # Extra renders to ensure content is properly displayed
        plotter.render()
        plotter.render()  # Second render often helps
        # Wait for rendering to complete
        import time
        time.sleep(0.2)  # Increased wait time
        # Render and save frame
        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        plotter.screenshot(frame_path)
    print(f"\nFrames saved to {frame_dir}")
    # Close the plotter
    plotter.close()
def create_video_from_frames(frames_dir, output_path, fps=30):
    """
    Create a video file from a directory of image frames using OpenCV.
    Parameters:
    - frames_dir: Directory containing the image frames (named frame_XXXX.png)
    - output_path: Path to save the output video file
    - fps: Frames per second for the video
    """
    import glob
    import re
    import cv2
    def numerical_sort(value):
        # Extract digits from filename for proper numerical sorting
        parts = re.compile(r'(\d+)').split(value)
        parts[1::2] = map(int, parts[1::2])  # Convert the extracted digits to integers
        return parts
    # add /frames to frames dir
    frames_dir = os.path.join(frames_dir, 'frames')
    # Get all frame paths and sort them numerically
    frame_paths = glob.glob(os.path.join(frames_dir, "frame_*.png"))
    frame_paths.sort(key=numerical_sort)
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return False
    print(f"Found {len(frame_paths)} frames to combine")
    try:
        # Read the first image to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            print(f"Could not read first frame: {frame_paths[0]}")
            return False
        height, width, layers = first_frame.shape
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        # add .mp4 to output path
        output_path = f"{output_path}/frames_video.mp4"
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        # Add all frames to video
        for i, frame_path in enumerate(frame_paths):
            print(f"Adding frame {i+1}/{len(frame_paths)}", end='\r')
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"\nWarning: Could not read frame {frame_path}, skipping")
                continue
            video.write(frame)
        # Release the video writer
        video.release()
        print(f"\nVideo saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        return False
# Example usage
if __name__ == "__main__":
    
    output_path = "/home/al/Documents/Motion_Detector/high_quality_results/output_insertion_200ep_newer"

    # Load the sequential images (rotate images 90° to the right after loading)
    volume_data = load_sequential_images(output_path, rotate_90_right=True)
    # Apply 3D Gaussian blur (uniform sigma for all dimensions)
    volume_data = apply_3d_gaussian_blur(volume_data, sigma=4.0) # default sigma of 2
    # Normalize and apply gamma correction
    volume_data = normalize_with_gamma(volume_data, gamma=0.05, min_clip=0.01, max_clip=0.99) # default gamma 0.05
    
    # Create gamma-corrected colorbars
    create_colorbars_with_labels(cmap="plasma", gamma=0.05, output_dir=output_path)
    
    # Option 1: Generate the frames

    create_rotating_volume_frames(
        volume_data,
        output_path=output_path,
        n_frames=180,  # 180 frames = 6 seconds at 30fps
        zoom_factor_z=5, # defaut z of 10
        title="",
        cmap="plasma",
        resolution=(1024, 1024)  # Full HD resolution
    )

    # Option 2: Create video from existing frames
    create_video_from_frames(
        frames_dir=output_path,
        output_path=output_path,
        fps=30
    )

    # Option 3: Visualize interactively
    visualize_volume(
        volume_data,
        zoom_factor_z=3, # default z of 10
        title="Alpha Layers Visualization",
        cmap="plasma"
    )
    