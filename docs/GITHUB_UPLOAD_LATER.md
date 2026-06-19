# GitHub Upload Later

GitHub upload was intentionally deferred. To publish later:

```bash
cd /home/al/Documents/CWDT_preprint_prep/code_release_local
python scripts/verify_release.py --data-root ../data_kaggle_package --sample-images 1
python scripts/build_release_manifest.py
git init
git config user.name "Al"
git config user.email "your-email@example.com"
git add .
git commit -m "Release CWDT figure reconstruction code"
git branch -M main
git remote add origin https://github.com/theMenonlab/diffusive_tomography.git
git push -u origin main
```

Before public upload, recheck:

- `python scripts/verify_release.py --data-root ../data_kaggle_package --sample-images 1` passes.
- `python scripts/run_paper_figure.py --figure 2 --data-root ../data_kaggle_package --dry-run` prints the expected command.
- No raw unpublished data beyond the intended tiny sample.
- No large `.pth`, `.pt`, `.npz`, raw image-stack, or output folders are staged for GitHub.
- `RELEASE_MANIFEST.csv` and `checksums_sha256.txt` have been regenerated after final edits.
- No local absolute paths are required in runnable examples unless clearly documented.
- License choice is still MIT for code and local hardware design files.
- Public Thingiverse/MakerWorld hardware model-page licenses match the intended repository license, or the README clearly states the dual-license situation.
- README points to the public Kaggle data package: <https://www.kaggle.com/datasets/alingold/continuous-wave-diffusive-tomography>.
