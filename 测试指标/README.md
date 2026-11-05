# 测试指标

本目录提供接缝局部质量评估代码，来源于原 Seam_composition。原目录被根 .gitignore 的 /Seam_composition/ 规则排除；这里保留独立源码副本，不包含模型权重、数据或结果。训练和常规测试说明见 [Seam/readme.md](../Seam/readme.md)。

## 三个 test 脚本

| 文件 | 用途 | 输入与输出 |
|---|---|---|
| test_2_before.py | 不进行逐样本迭代优化，评估预训练模型的初始拼接效果，作为优化前基线 | --test_path 测试集，--result_path 图像输出目录，--output_file 指标文本；输出各样本及平均指标、接缝/变形/掩码等可视化 |
| test_2_iter_one.py | 对一对图像进行测试时迭代优化，观察接缝损失及质量变化 | --path 图像目录（末尾加 /），--img1_name、--img2_name、--max_iter（默认50）；输出迭代图像、终端指标、损失历史文件，并打印耗时和显存 |
| test_2_iter_mul.py | 按排序遍历 input1/input2 中的 JPG 图像对，每对重新加载模型并优化，汇总指标 | --zpath 包含 input1 和 input2 的目录，--max_iter；输出路径和指标文本另有硬编码，运行前必须修改 |

这些 test 是 GPU 推理/测试时优化和指标评估程序，不是 pytest 单元测试，也不是多机分布式训练。

## 指标含义

在接缝上的有效采样点提取局部 patch，对两幅变形图像的 patch 乘以共同有效掩码，再计算并平均以下指标；边缘无法取得完整 patch 的位置跳过。具体调用的 patch_size 请以各脚本调用处为准（单对优化脚本评估处使用35）。

- ZNCC：代码实际返回并打印的是误差 Q = 1 - (ZNCC + 1) / 2，越低越好。虽然终端标签写 ZNCC，它并非原始相关系数。
- RMSE：局部像素均方根误差，越低越好。
- PSNR：按峰值255计算，越高越好，单位 dB。
- SSIM：data_range=255 的结构相似性，越高越好。

这里比较的是接缝附近两幅变形图像的一致性，不是与真值全景图对比的全图质量。掩码外像素置零后仍参与 patch 统计；不要与仅在有效像素上统计的实现直接混用。确认评估输入为0~255范围，否则PSNR和SSIM口径不一致。

## 运行前必须配置

1. 使用原项目环境，安装匹配的 PyTorch/torchvision、NumPy、OpenCV、scikit-image，以及 network.py / utils 引用的依赖。需要 NVIDIA CUDA 环境；脚本含显式 .cuda() 调用。
2. 三个脚本的 model_path 都硬编码为 /amax/home/taoyz/Seam/Seam_composition/model/epoch100_model.pth。下载权重后改为实际绝对路径；本目录不附权重。迭代脚本还读取 checkpoint 的 optimizer 和 epoch。
3. 修改 test_2_iter_mul.py 主入口中 args.output_file 和 testpath；仅传 --zpath 不会改变这两个输出位置。确认两个输入目录文件数量一致且排序后一一对应。
4. 输出目录提前创建，路径参数末尾保留 /，因为部分输出采用字符串拼接。
5. 从本目录运行，以便解析 network、dataset、loss、grid_res、utils 的同级导入。源脚本使用 os.path.dirname("__file__") 字符串而非 __file__ 变量，MODEL_DIR 依赖工作目录，不应认为可从任意目录启动。

示例（先替换代码中的权重路径和上述硬编码输出路径）：

```bash
cd 测试指标
mkdir -p /path/to/results
python test_2_before.py --gpu 0 --batch_size 1 --test_path /path/to/test/ --result_path /path/to/results/ --output_file /path/to/results/metrics.txt
python test_2_iter_one.py --gpu 0 --path /path/to/pair/ --img1_name 01.jpg --img2_name 02.jpg --max_iter 50
python test_2_iter_mul.py --gpu 0 --zpath /path/to/dataset/ --max_iter 50
```

## 已知行为与限制

- 单对脚本虽然打印 earlystop，但对应 break 被注释，不能据日志判断已真正提前终止。
- 当没有可用接缝 patch 时 evaluate_seam_quality 返回 None，部分调用直接解包，可能报错。
- 单对脚本在 CUDA 分支将输入缩小为原图一半，多对脚本没有相同预处理，比较两种入口结果时要统一尺度。
- 保留原始实验逻辑，本次只整理发布目录和说明，没有运行 GPU 实验或验证论文指标。

## 支持代码

network.py：网络和变形/拼接输出；loss.py：测试时优化损失；dataset.py：数据读取；grid_res.py：网格分辨率；utils/：几何变换等辅助实现。
