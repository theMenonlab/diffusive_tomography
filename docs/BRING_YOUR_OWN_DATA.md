# Bring Your Own Data

The current reconstruction code expects an ordered stack of images captured while the illumination fiber is scanned across the sample.

## Input Folder

Create a folder containing only the images for one scan:

```text
my_scan/
  IMG_0001.png
  IMG_0002.png
  IMG_0003.png
  ...
```

The scripts sort filenames alphabetically, so use names that preserve acquisition order. PNG, JPG, JPEG, TIF, and TIFF inputs are the expected formats.

## Start from a Paper Config

Copy the closest paper config and edit the paths:

```bash
cp configs/figure4_fungus.json my_scan_config.json
```

Edit:

- `data_paths.folder_path`: input image folder.
- `data_paths.output_base_path`: output folder.
- `image_cropping`: crop boundaries and rotation needed to align the sample.
- `physical_parameters.img_x_um`: field of view width after cropping/resizing.
- `physical_parameters.sample_z_um`: modeled depth.
- `light_source`: scan spacing, direction, center image, source depth, NA, and blur.
- `training.num_epochs`: use a small value for smoke tests, then increase for final reconstructions.

## Run

```bash
python src/forward_model.py my_scan_config.json
```

The main outputs are:

- `alpha_layers/`: learned attenuation layers.
- `alpha_inv_layers/`: inverted visualization of attenuation layers.
- `predicted/`: model-predicted surface images.
- `incident/`: modeled incident illumination.
- `ground_truth/`: processed target images.
- `loss_log.txt`: training loss by epoch.

## Practical Checks

Before a full run, use a short 1-5 epoch config and confirm:

- the image stack loads in the intended order;
- the crop/rotation captures the same field of view in every frame;
- the source direction and `center_image` match the scan;
- `predicted/` and `ground_truth/` have the expected shape;
- the loss decreases rather than immediately becoming NaN.

For new datasets, keep the raw input stack, the exact JSON config, the script version, and the generated `loss_log.txt` together. That is the minimum needed for someone else to rerun or audit the reconstruction.
