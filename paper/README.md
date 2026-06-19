# Paper Figure Reproduction

This folder pins the exact reconstruction scripts and JSON configurations used for the paper Figure 2-5 model outputs. The large input image stacks and saved reference outputs are not stored here; they are in the companion `data_kaggle_package`.

## Figure Map

| Figure | Folder | Input images | Epochs | Saved result source in `Motion_Detector` |
| --- | --- | ---: | ---: | --- |
| 2 | `figure_2_transmission` | 53 | 200 | `high_quality_results/output_tartrazine_transmission_200epochs` |
| 3 | `figure_3_phantom` | 24 | 50 | `high_quality_results/output_20250813_thick_phantom` |
| 4 | `figure_4_fungus` | 21 | 200 | `high_quality_results/output_fungus_200ep` |
| 5 | `figure_5_insertion` | 27 | 200 | `high_quality_results/output_insertion_200ep_newer` |

## Recommended Command

From the repository root:

```bash
python scripts/verify_release.py --data-root ../data_kaggle_package --sample-images 1
python scripts/run_paper_figure.py --figure 2 --data-root ../data_kaggle_package --dry-run
python scripts/run_paper_figure.py --figure 2 --data-root ../data_kaggle_package
```

Use `--figure 3`, `--figure 4`, or `--figure 5` for the other paper reconstructions.

## Direct Command

The exact scripts expect `input_images` to be relative to the current working directory. If bypassing the runner, first change into the matching data-package figure folder:

```bash
cd /path/to/data_kaggle_package/figures/figure_2_transmission
python /path/to/diffusive_tomography/paper/figure_2_transmission/forward_model_exact.py config_package.json
```

Outputs are written to `reproduction_output/`.
