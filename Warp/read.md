# Warp: Training and Before/Iter Outputs

See [readme.md](readme.md) for environment, dataset, training, and model download instructions.

## Script Roles

| Script | Purpose | Numerical metrics |
|---|---|---|
| `train.py` | Train the Warp network on paired images. | Training losses. |
| `test_2_before.py` | Run the pre-trained network once and compose a single image pair using graph cut. Network parameters are not updated. | Not computed. |
| `test_2_iter.py` | Optimize the network on a single test pair using a seam-guided loss and save the iterative result. | Not computed. |

Both test scripts use `Network` and `build_new_ft_model` from the local `network.py`. They generate before/iter images; use the scripts in [../Seam_composition/read.md](../Seam_composition/read.md) for metric evaluation. The composition helpers differ between the two directories, so their outputs should not be mixed without checking mask semantics.

## Input and Model

Place both input images in one directory. The `--path` argument must end with `/`; outputs are saved in that directory.

The configured checkpoint path is:

```text
/amax/home/taoyz/Seam/Warp/model/epoch100_model.pth
```

Update it in both scripts if needed. Currently, checkpoint loading is also conditional on finding a `.pth` file in `MODEL_DIR`. `MODEL_DIR` is derived from the working directory's parent because the code uses the literal string `"__file__"`. Consequently, placing a checkpoint at the configured path alone does not guarantee it will be loaded. Verify that the script reports loading the intended model before using its results.

## Before: Direct Inference

Run from this directory:

```bash
python test_2_before.py --gpu 0 --path /path/to/pair/ --img1_name 01.jpg --img2_name 02.jpg
```

Outputs include `before_optimization_cut.jpg`, `before_seam_cut.jpg`, warped images, and masks. Here, “before” means before test-time parameter optimization; graph-cut composition is still performed.

## Iterative Optimization

```bash
python test_2_iter.py --gpu 0 --path /path/to/pair/ --img1_name 01.jpg --img2_name 02.jpg --max_iter 50
```

Outputs include `edge_iter-XXX.jpg` and warped images. The current script has early stopping, so `--max_iter 50` means at most 50 updates. It also renders using meshes computed before the current optimizer update. To report exactly 50 updates, disable early stopping and recompute predictions after the final update.

## Comparison Protocol

Use the same checkpoint, input pair, resolution, composition function, and metric implementation for before/after comparisons. The current iterative script's first saved result uses a training-mode forward pass and should not be treated as an identical replacement for the evaluation-mode direct-inference baseline.
