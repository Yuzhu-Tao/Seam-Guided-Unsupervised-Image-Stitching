# Seam Composition and Metric Evaluation

This directory provides a direct-inference baseline and seam-guided test-time optimization. All scripts use the same local `Network`; the iterative scripts update its parameters separately for each test pair.

## Script Roles

| Script | Input | Behavior | Metrics |
|---|---|---|---|
| `test_2_before.py` | Dataset with `input1/` and `input2/`. | Direct inference in evaluation mode; no parameter updates. | Per-pair metrics and dataset means. |
| `test_2_iter_one.py` | One image pair in a shared directory. | Test-time optimization; default 50 iterations. | Prints single-pair metrics. |
| `test_2_iter_mul.py` | Dataset with `input1/` and `input2/`. | Reloads the checkpoint and optimizes each pair; default maximum 50 iterations. | Per-pair metrics and dataset means. |

For dataset-level before/after evaluation, use `test_2_before.py` and `test_2_iter_mul.py` after addressing the implementation notes below. The two test scripts in `../Warp/` save images but do not compute these numerical metrics.

## Pre-trained Model

Both stages use the same pre-trained Warp checkpoint, `epoch100_model.pth`, available at [Baidu Cloud](https://pan.baidu.com/s/1ngBykNcCrAWJbEhWTDa3ZQ?pwd=8ppg). Extraction code: `8ppg`.

The configured checkpoint path for all three scripts is:

```text
/amax/home/taoyz/Seam/Seam_composition/model/epoch100_model.pth
```

Place the model there or update the scripts to match your installation.

## Dataset Layout

```text
testing/
├── input1/
│   ├── 000001.jpg
│   └── ...
└── input2/
    ├── 000001.jpg
    └── ...
```

Images are paired by sorted filenames. Use matching filenames and equal pair counts. Use batch size 1 for the baseline evaluation.

## Before: Direct Inference and Metrics

Run from this directory:

```bash
python test_2_before.py --gpu 0 --batch_size 1 --test_path /path/to/testing/ --result_path /path/to/results/before/ --output_file /path/to/results/before_metrics.txt
```

Create the metrics output parent directory first. Keep a trailing `/` on `--result_path`. The script saves stitched images, seam visualizations, warps, and masks, and appends per-pair and average metrics to the output file.

## Iterative Evaluation on a Dataset

```bash
python test_2_iter_mul.py --gpu 0 --zpath /path/to/testing/ --max_iter 50
```

Currently, `testpath` and `args.output_file` are hard-coded in the script's main block. Update both before running. Each pair starts from the checkpoint, rather than inheriting the previous pair's optimized parameters.

## Iterative Evaluation on One Pair

```bash
python test_2_iter_one.py --gpu 0 --path /path/to/pair/ --img1_name 01.jpg --img2_name 02.jpg --max_iter 50
```

Keep a trailing `/` on `--path`. Images and outputs share that directory. This script currently halves the input image resolution before processing; remove or standardize that preprocessing before comparing with full-resolution dataset results.

## Metric Definitions

Metrics compare the two warped images in 35-by-35 patches centered on seam pixels. They measure local cross-image consistency, rather than reconstruction against a ground-truth panorama.

| Reported metric | Definition | Preferred direction |
|---|---|---|
| ZNCC error | `(1 - ZNCC) / 2`, averaged across patches. The scripts label it as ZNCC. | Lower. |
| RMSE | Per-patch root mean squared error, averaged across patches. | Lower. |
| PSNR | Per-patch PSNR with intensity range 255, averaged across patches. | Higher. |
| SSIM | Per-patch SSIM with intensity range 255, averaged across patches. | Higher. |

The four primitive metric functions are identical across the three scripts. Dataset means weight each successfully evaluated image pair equally.

## Current Implementation Notes

- The iterative scripts import `cal_lp_loss2_cut`, but the local `loss.py` currently has no active definition of that function. Remove that unused import or restore its definition before running.
- `test_2_iter_mul.py` has early stopping. Its `--max_iter 50` is an upper limit, not a guarantee of exactly 50 updates.
- Iterative outputs use meshes computed before the current optimizer update. Recompute predictions after the final update to evaluate the actual 50-update model.
- Iterative scripts use training mode. For a direct-inference baseline and final assessment, explicitly use evaluation mode and a fresh forward pass under `torch.no_grad()`.
- Loading the checkpoint optimizer state overrides the optimizer's initially configured learning rate and restores its momentum. Record or standardize the optimizer initialization for reproducible experiments.
- Metrics currently multiply both patches by the common-validity mask and then include the zeroed invalid pixels in the statistics. This can bias scores. Valid-pixel statistics and an explicit valid-window rule for SSIM are needed for reliable masked evaluation.
- Failed pairs may be skipped independently in before and after evaluation. Report failures and compare averages over the same successful pair set.
- Each result is evaluated along its own seam. This measures the quality of each selected seam; it is not a fixed-location alignment comparison.

For a strict comparison, load the same checkpoint for each pair, evaluate and save the baseline, perform exactly 50 updates without early stopping, then run a fresh evaluation-mode prediction and compute the final metrics with the same preprocessing and validity rules.
