# 测试指标

Compute seam-local quality metrics before and after test-time optimization. See [Seam/readme.md](../Seam/readme.md) for environment setup, datasets, and the pre-trained model.

## Scripts

| Script | Purpose |
|---|---|
| `test_2_before.py` | Evaluate image pairs using the pre-trained model without iterative optimization; save results and average metrics. |
| `test_2_iter_one.py` | Optimize a single image pair and report seam-quality metrics, iteration time, and GPU memory usage. |
| `test_2_iter_mul.py` | Optimize multiple image pairs from paired `input1/` and `input2/` folders and report per-pair and average metrics. |

## Pre-trained Model

All three test scripts use the same pre-trained model as `Seam/`: `epoch100_model.pth`. No separate evaluation model is required. See [Seam: Pre-trained Model](../Seam/readme.md#pre-trained-model) for the download link.

Set `model_path` in all three scripts to this checkpoint. The before-optimization script performs direct inference; the iterative scripts start from the same checkpoint and refine the network for each image pair. Keep the complete checkpoint, including optimizer state, for iterative optimization.

## Setup

Use the same environment as `Seam/`, with `scikit-image` installed. Update `model_path` in all three scripts to the downloaded `epoch100_model.pth`. Iterative scripts also require the checkpoint optimizer state.

Run all commands from this directory:

```bash
cd 测试指标
```

## Before Optimization

Prepare paired `input1/` and `input2/` folders and create the output directory.

```bash
mkdir -p /path/to/results
python test_2_before.py --gpu 0 --batch_size 1 --test_path /path/to/test/ --result_path /path/to/results/ --output_file /path/to/results/metrics.txt
```

## Single-Pair Optimization

Place two images in the same directory. The path must end with `/`.

```bash
python test_2_iter_one.py --gpu 0 --path /path/to/pair/ --img1_name 01.jpg --img2_name 02.jpg --max_iter 50
```

Results are saved in the image directory; metrics are printed to the console. This version reports stable-loss events but does not break early because its `break` is commented out.

## Multiple-Pair Optimization

Prepare matching JPG files in `input1/` and `input2/`; files are paired by sorted filenames. Update `args.output_file` and `testpath` in the script to your metrics and image-output paths before running.

```bash
python test_2_iter_mul.py --gpu 0 --zpath /path/to/dataset/ --max_iter 50
```

## Metrics

Metrics compare local patches around the seam in the two warped images, after applying the overlap mask.

| Console label | Meaning | Better |
|---|---|---|
| ZNCC | Converted error: `1 - (ZNCC + 1) / 2`, not raw correlation | Lower |
| RMSE | Root mean squared pixel error | Lower |
| PSNR | Peak signal-to-noise ratio, using peak value 255 | Higher |
| SSIM | Structural similarity, using data range 255 | Higher |
