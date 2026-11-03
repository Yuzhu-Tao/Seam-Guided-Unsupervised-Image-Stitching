# Stage 1: Warp

This stage trains the image warping network with parallax-aware mask generation.

## Dataset

Download **input-data-UDIS.zip** from [Baidu Netdisk](https://pan.baidu.com/s/1AMI9FvWeYj4w9axpEtCCwg?pwd=whuf).

Extraction code: `whuf`.

Extract the dataset and locate the training directory containing paired JPEG images:

```text
training/
├── input1/
│   ├── 000001.jpg
│   └── ...
└── input2/
    ├── 000001.jpg
    └── ...
```

Images in the two folders are paired by their sorted filenames.

## Training

Activate the `seam` environment and run the following command from this directory:

```bash
python train.py
```

The default training path is `/amax/home/taoyz/new_final_2/training/`. Override it for your dataset location:

```bash
python train.py --train_path /path/to/training/ --gpu 0
```

The default batch size is `4`, and training runs for `100` epochs. You can change these settings with `--batch_size` and `--max_epoch`.

Checkpoints and TensorBoard logs are saved in `../model_new_final_2_crop_edge_loss_bn/` and `../summary_new_final_2_crop_edge_loss_bn/`, respectively.

## Test on New_datasets

The pre-trained model of Warp (`epoch100_model.pth`) is available at [Baidu Cloud](https://pan.baidu.com/s/1ngBykNcCrAWJbEhWTDa3ZQ?pwd=8ppg).

Extraction code: `8ppg`.

## Before/Iter Outputs and Evaluation

See [read.md](read.md) for test script roles, commands, and comparison limitations.
