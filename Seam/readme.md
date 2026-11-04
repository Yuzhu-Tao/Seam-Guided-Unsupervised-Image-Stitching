# Seam

## Scripts

| Script | Purpose |
|---|---|
| `train.py` | Train the image warping network with parallax-aware mask generation on paired training images. |
| `test_2_before.py` | Stitch a single image pair using the pre-trained model and graph-cut composition, before test-time optimization. Network parameters are not updated. |
| `test_2_iter.py` | Refine the network on a single image pair using a seam-guided loss, then save the stitching result. Network parameters are updated for the current pair. |

The framework first trains the warping network, then refines alignment on each test image pair through seam-guided optimization. The default iteration limit is 50, with early stopping when the loss stabilizes.

## Dataset

Train: [Baidu Netdisk](https://pan.baidu.com/s/1kI0yxoYp89yzWXGN7PdN1g?pwd=q4ff), code: `q4ff`. File: `train.zip`.

Test: [Baidu Netdisk](https://pan.baidu.com/s/1135E4umtqJC1iw4JW53jvw?pwd=pa7h), code: `pa7h`. File: `testing.zip`.

Extract the training archive and locate the directory containing the paired image folders:

```text
/path/to/training/
├── input1/
│   ├── 000001.jpg
│   └── ...
└── input2/
    ├── 000001.jpg
    └── ...
```

Set `--train_path` to this directory. JPEG images in `input1/` and `input2/` are paired by sorted filenames, so keep the two folders' image counts and filename ordering consistent.

For either test script, select one corresponding image pair from the extracted test dataset and place both images in a single directory with distinct names, such as `01.jpg` and `02.jpg`. Pass that directory with `--path` and the filenames with `--img1_name` and `--img2_name`.

## Pre-trained Model

Download `epoch100_model.pth` from [Baidu Netdisk](https://pan.baidu.com/s/1OU5Ihqp5knEtrPh45af7bA?pwd=pay1), code: `pay1`.

Both test scripts start from this model: `test_2_before.py` performs direct inference, while `test_2_iter.py` refines the model on the current image pair. See [Model Configuration](#model-configuration) below for the required checkpoint path setup.

## Setup

Run from the repository root:

```bash
conda env create -f requirement.yml
conda activate seam
cd Seam
```

## Training: train.py

Extract the training dataset into a directory containing paired `input1/` and `input2/` folders. Images are paired by their sorted filenames.

```bash
python train.py --train_path /path/to/training/ --gpu 0
```

The default batch size is `4`, and training runs for `100` epochs. Use `--batch_size` and `--max_epoch` to change them. Checkpoints and TensorBoard logs are saved in `../model/` and `../summary/`, respectively.

## Testing: test_2_before.py

Use this script to obtain the stitching result before iterative optimization. It predicts the image transformation once, searches for a seam, and composes the two images using graph cut.

Place both images in the same directory, then run:

```bash
python test_2_before.py --gpu 0 --path /path/to/pair/ --img1_name 01.jpg --img2_name 02.jpg
```

The `--path` argument must end with `/`. Outputs are saved in that directory, including:

- `before_optimization_cut.jpg`: the stitched image before iterative optimization.
- `before_seam_cut.jpg`: a visualization marking the seam.
- Warped images, predicted masks, and graph-cut selection masks.

## Iterative Optimization: test_2_iter.py

Use this script to refine stitching for the current image pair. Each iteration predicts the transformation, finds a seam, and updates the network using color and gradient alignment losses around the seam. Optimization runs on images resized to `512 × 512`; output images are rendered at the input image scale.

```bash
python test_2_iter.py --gpu 0 --path /path/to/pair/ --img1_name 01.jpg --img2_name 02.jpg --max_iter 50
```

Outputs are saved in the input directory, including:

- `edge_iter-XXX.jpg`: the stitching result at the stopping iteration.
- `edge_seam_cut.jpg`: a visualization marking the resulting seam.
- Initial results, warped images, and masks.

`--max_iter` sets the maximum number of iterations. The script stops early when the loss stabilizes. It saves images rather than a new model checkpoint. Both test scripts produce visual results without computing numerical quality metrics.

## Model Configuration

Place `epoch100_model.pth` in `../model/` and update `model_path` in both test scripts to `../model/epoch100_model.pth`. Run the scripts from `Seam/` and confirm that the console reports loading the model. Keep the complete checkpoint, including optimizer state, for iterative optimization.
