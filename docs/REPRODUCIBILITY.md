# Reproducing Figures 2-5

This repository is paired with a Kaggle dataset containing the input stacks and saved reference outputs:

```text
https://www.kaggle.com/datasets/alingold/continuous-wave-diffusive-tomography
```

The local package path during preparation was:

```text
/home/al/Documents/CWDT_preprint_prep/data_kaggle_package
```

The original working tree audited for provenance was:

```text
/home/al/Documents/Motion_Detector
```

## Provenance Table

| Figure | Data-package folder | Original result folder | Exact script SHA-256 | Images | Layers | Epochs |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 2 | `figures/figure_2_transmission` | `high_quality_results/output_tartrazine_transmission_200epochs` | `c51fba093b2b96f731d34f72cd7205ae7c5497a780b853ae6872d9d3b4c00f35` | 53 | 16 | 200 |
| 3 | `figures/figure_3_phantom` | `high_quality_results/output_20250813_thick_phantom` | `612695c2f59224155e81f6aa7e0f8d962842f850699cc450a12ff242d0b8ab12` | 24 | 16 | 50 |
| 4 | `figures/figure_4_fungus` | `high_quality_results/output_fungus_200ep` | `ea76390752c5b26ac942d578529dedec3204c2ab0322756015f6b1d6570c8d9a` | 21 | 16 | 200 |
| 5 | `figures/figure_5_insertion` | `high_quality_results/output_insertion_200ep_newer` | `2256b423f14faead09d7e65beac886715b0ee38e3f0908a8998653234095322b` | 27 | 16 | 200 |

The copied `paper/figure_*/forward_model_exact.py` files were byte-for-byte matched against the corresponding `Motion_Detector/high_quality_results/.../forward_model.py` files.

## Setup

```bash
cd /path/to/diffusive_tomography
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A CUDA-capable PyTorch install is recommended for the full figure reconstructions. CPU runs should work for inspection and small tests, but the full 50-200 epoch runs can be slow.

## Verify Package Integrity

```bash
cd /path/to/diffusive_tomography
python scripts/verify_release.py --data-root /path/to/data_kaggle_package --sample-images 1
```

This checks the exact-script hashes, portable config hashes, expected input-image counts, saved reference outputs, and whether sample images can be opened.

## Run a Paper Figure

Use the runner from the repository root:

```bash
python scripts/run_paper_figure.py --figure 2 --data-root /path/to/data_kaggle_package --dry-run
python scripts/run_paper_figure.py --figure 2 --data-root /path/to/data_kaggle_package
```

The runner changes into the matching data-package figure directory and launches the exact script with `config_package.json`. This matters because the original scripts resolve `input_images` relative to the current working directory.

Outputs are written under:

```text
/path/to/data_kaggle_package/figures/figure_X/reproduction_output
```

The saved paper-reference outputs remain in:

```text
/path/to/data_kaggle_package/figures/figure_X/results
```

## Direct Script Usage

The data package is also self-contained:

```bash
cd /path/to/data_kaggle_package/figures/figure_2_transmission
python code/forward_model_exact.py config_package.json
```

or using the GitHub copy of the exact script:

```bash
cd /path/to/data_kaggle_package/figures/figure_2_transmission
python /path/to/diffusive_tomography/paper/figure_2_transmission/forward_model_exact.py config_package.json
```

## Determinism Notes

The exact scripts and configs are pinned, but GPU libraries and PyTorch versions can introduce small numerical differences. Treat the saved `results/` folders as reference outputs for scientific inspection rather than assuming every regenerated PNG will be byte-identical on every machine.

No `.pth` checkpoint is required for Figures 2-5. These reconstructions optimize the attenuation-layer representation from each image stack and save the learned layers as output images. The only `.pth` file found in the audited `Motion_Detector` tree was a separate 374 MB U-Net checkpoint from `output_20250815_u-net_tartrazine_transmission`, which is not needed for reproducing the paper Figure 2-5 forward-model results.
