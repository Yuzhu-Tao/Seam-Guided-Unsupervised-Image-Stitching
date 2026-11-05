<h1 align="center">Seam-Guided Unsupervised Image Stitching With Parallax-Aware Mask Generation</h1>

<p align="center">
  <img src="img.png" alt="Overview of parallax-aware warping and seam-guided alignment" width="100%">
</p>

The framework consists of two stages:

1. **Warp training:** Learn parallax-aware masks to guide image warping while preserving natural content.
2. **Seam-guided alignment:** Refine local alignment around the estimated seam for up to 50 iterations, with early stopping when the loss stabilizes.

See [Seam/readme.md](Seam/readme.md) for dataset and pre-trained model downloads, environment setup, and training and testing instructions.

## Evaluation / 测试指标

For seam-quality metrics and the three evaluation scripts, see [测试指标/README.md](测试指标/README.md). 测试指标及三个 test 脚本的用途、配置和运行方式，请进入该目录查看。

## Citation

```bibtex
@article{tao2026seam,
  title={Seam-Guided Unsupervised Image Stitching with Parallax-Aware Mask Generation},
  author={Tao, Yuzhu and Nie, Lang and Chang, Yakun and Li, Bo and Li, Qingwen and Li, Hong and Wei, Shikui},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

## Contact

We welcome questions, feedback, and discussions about our work. Please feel free to contact Yuzhu Tao at [taoyuzhu@bjtu.edu.cn](mailto:taoyuzhu@bjtu.edu.cn).

Thank you for your interest in our work!
