# Diffusive Tomography CWDT Code Release

This repository accompanies the preprint draft:

**Low-Cost Continuous-Wave Diffusive Microtomography with Fiber-Scanned White-Light Illumination**

Public repository target: <https://github.com/theMenonlab/diffusive_tomography>

The repository contains the CWDT reconstruction code, exact Figure 2-5 reproduction scripts/configs, release-verification utilities, a tiny sample image folder, and the printable TriAxis micropositioner design files used to hold and scan the illumination fiber.

The full Figure 2-5 image stacks and saved reference outputs are packaged separately on Kaggle: <https://www.kaggle.com/datasets/alingold/continuous-wave-diffusive-tomography>. They are too large for normal GitHub storage.

## Contents

- `src/forward_model.py` - current reusable RGB CWDT forward-model reconstruction script.
- `src/fourier_resolution.py` - Fourier-based layer sharpness/resolution analysis helper.
- `src/depth_resolution_postprocess_v3.py` - postprocessing helper for depth-resolution outputs.
- `src/volume_viewer_save_vis.py` - volume rendering/rotation helper.
- `paper/figure_*/` - exact scripts and configs copied from the saved Figure 2-5 result folders.
- `configs/figure2_transmission.json` - tartrazine-cleared poplar section model configuration.
- `configs/figure3_phantom.json` - scattering phantom model configuration.
- `configs/figure4_fungus.json` - fungal/root sample model configuration.
- `configs/figure5_insertion.json` - inserted-fiber poplar branch model configuration.
- `scripts/run_paper_figure.py` - launches an exact paper-figure reproduction from the Kaggle-style data package.
- `scripts/verify_release.py` - checks exact-script hashes, configs, expected input counts, and optional sample image loading.
- `examples/tiny_sample/` - three example images for smoke testing file loading.
- `hardware/micropositioner/` - STL files and the Bambu/3MF print project for the TriAxis micropositioner.
- `docs/` - reproducibility, data-placement, GitHub upload, and bring-your-own-data notes.

## Figure 2-5 Reproduction

The exact scripts used for the saved Figure 2-5 model outputs are preserved in this repository under `paper/figure_*/` and also in the Kaggle-ready data package under each figure's `code/` folder.

Verify the local release and data package:

```bash
cd /path/to/diffusive_tomography
python scripts/verify_release.py --data-root /path/to/data_kaggle_package --sample-images 1
```

Run a dry run for a figure:

```bash
python scripts/run_paper_figure.py --figure 2 --data-root /path/to/data_kaggle_package --dry-run
```

Launch the full reconstruction:

```bash
python scripts/run_paper_figure.py --figure 2 --data-root /path/to/data_kaggle_package
```

Use `--figure 3`, `--figure 4`, or `--figure 5` for the other reconstructions. Outputs are written to `reproduction_output/` inside the matching data-package figure folder.

See `docs/REPRODUCIBILITY.md` for provenance, hashes, and direct commands.

## Environment

Tested locally with Python 3 and CUDA-capable PyTorch. CPU execution should work for small smoke tests but full figure reconstructions are expected to be slow without a GPU.

Install the core Python dependencies:

```bash
python -m pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch build, install the matching `torch` and `torchvision` wheels first using the instructions for your CUDA version, then install the remaining requirements.

## Running on New Data

Copy or edit one of the JSON config files so that:

- `data_paths.folder_path` points to a local input image folder.
- `data_paths.output_base_path` points to a writable output folder.
- `interface.use_gui_default` is `true` for noninteractive runs.

Then run:

```bash
cd /path/to/code_release_local
python src/forward_model.py configs/figure2_transmission.json
```

For new datasets, see `docs/BRING_YOUR_OWN_DATA.md`.

## Data and Model Placement

GitHub should contain code, exact configs, small examples, docs, and hardware files. The Kaggle dataset contains the full Figure 2-5 input stacks and saved reference outputs. See `docs/DATA_AND_MODEL_PLACEMENT.md`.

No `.pth` checkpoint is required for reproducing Figures 2-5. The forward-model scripts optimize the attenuation-layer representation directly from the image stack and save the trained layers as output images.

## Hardware Files

The TriAxis micropositioner files are in `hardware/micropositioner/`.
Public mirrors and editable CAD are available here:

- MakerWorld: <https://makerworld.com/en/models/1256154-triaxis-micropositioner>
- Thingiverse: <https://www.thingiverse.com/thing:6976184>
- Onshape CAD: <https://cad.onshape.com/documents/c31396d958296393a0cbb499/w/10e122b0be94b2b2a2f3787d/e/975a9698867d3d77a59824b4>

The local files are prepared for release under the repository MIT license. The existing Thingiverse archive metadata reports Creative Commons Attribution-ShareAlike, so harmonize the public model-page licenses before final upload if a single license across all mirrors is important.

## Notes

The saved figure models are proof-of-concept reconstructions from intensity-only continuous-wave image stacks. They reproduce measured surface images and produce qualitative attenuation volumes, but they are not independently validated quantitative depth maps.
