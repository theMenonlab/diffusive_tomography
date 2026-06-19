# Data and Model Placement

The release is split between GitHub and the Kaggle data package:

```text
https://www.kaggle.com/datasets/alingold/continuous-wave-diffusive-tomography
```

## Keep in GitHub

- Reconstruction source code in `src/`.
- Exact paper figure scripts/configs in `paper/figure_*/`.
- Runner and verification utilities in `scripts/`.
- Tiny example images in `examples/tiny_sample/`.
- TriAxis micropositioner hardware files in `hardware/micropositioner/`.
- Documentation, license, checksums, and release manifest.

The GitHub folder is about 50 MB before final manifest generation, mostly because the micropositioner STL and 3MF files are included.

## Keep in Kaggle

- Full Figure 2-5 input image stacks.
- Saved paper-reference outputs in each `results/` folder.
- Composite `paper_figure.png` files.
- Data-package manifest and SHA-256 checksums.

Current local figure-package sizes:

| Figure | Folder | Approx. size | Input images | Saved result files |
| --- | --- | ---: | ---: | ---: |
| 2 | `figure_2_transmission` | 577 MB | 53 | 193 |
| 3 | `figure_3_phantom` | 306 MB | 24 | 82 |
| 4 | `figure_4_fungus` | 334 MB | 21 | 135 |
| 5 | `figure_5_insertion` | 279 MB | 27 | 183 |

The combined data package is about 1.4 GB, which is not appropriate for normal GitHub storage.

## Model Files

The Figure 2-5 physics reconstructions do not use a saved neural-network checkpoint. A run retrains the layer attenuation representation from the input stack and writes `alpha_layers/`, `alpha_inv_layers/`, `predicted/`, `incident/`, and related outputs.

The only large checkpoint found in `/home/al/Documents/Motion_Detector` was:

```text
output_20250815_u-net_tartrazine_transmission/best_model.pth
```

That file is about 374 MB and belongs to a separate U-Net experiment, not the main Figure 2-5 forward-model results. Do not put it in GitHub unless it becomes a documented supplemental model. If included later, put it in Kaggle, Zenodo, or a GitHub release asset rather than the main repository history.

## Final Link

The manuscript and GitHub README should point to the Kaggle URL above. Add a DOI later if Kaggle or another archive assigns one.
