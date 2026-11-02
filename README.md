# Warp

Image warping code derived from UDIS2 / UDIS++.

## Source files

- `train.py`: training entry point.
- `network.py`: network and warping routines.
- `loss.py`: loss functions.
- `dataset.py`: dataset loading.
- `grid_res.py`: mesh resolution.
- `utils/`: DLT, homography, and TPS utilities.

This is a source-code snapshot. Dataset paths and checkpoint paths still require configuration. Training and inference have not been validated for this standalone layout.

## Attribution

Based on [UDIS2](https://github.com/nie-lang/UDIS2), Parallax-Tolerant Unsupervised Deep Image Stitching (ICCV 2023). The existing Apache-2.0 license is preserved in `LICENSE`.
