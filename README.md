
# 🚀 CIFAR-10 图像分类项目

基于 PyTorch 实现的自定义卷积神经网络，在 CIFAR-10 测试集上准确率约 **86%**。  
项目包含模型训练脚本、预训练权重以及一个支持拍照/上传的 Gradio 网页预测界面。

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🧠 网络结构

- **特征提取**：3 个卷积块，每个块包含两个 3×3 卷积层 + BatchNorm + ReLU
- **通道变化**：32 → 64 → 128
- **池化策略**：前两个卷积块后分别使用 2×2 MaxPool
- **分类头**：AdaptiveAvgPool2d(8×8) + Flatten + Linear(8192→64) + ReLU + Dropout(0.5) + Linear(64→10)
- **激活函数**：ReLU
- **训练策略**：50 轮训练并保存测试准确率最高时的模型权重

在 CIFAR-10 测试集上的最佳分类准确率约为 **86%**。

### 📊 网络结构总览

| 层级 | 模块 | 输出大小 |
| --- | --- | --- |
| 输入 | - | 3×32×32 |
| Block 1 | Conv2d(3→32, k=3, p=1) + BN + ReLU<br>Conv2d(32→32, k=3, p=1) + BN + ReLU | 32×32×32 |
| Pool 1 | MaxPool2d(2×2) | 32×16×16 |
| Block 2 | Conv2d(32→64, k=3, p=1) + BN + ReLU<br>Conv2d(64→64, k=3, p=1) + BN + ReLU | 64×16×16 |
| Pool 2 | MaxPool2d(2×2) | 64×8×8 |
| Block 3 | Conv2d(64→128, k=3, p=1) + BN + ReLU<br>Conv2d(128→128, k=3, p=1) + BN + ReLU | 128×8×8 |
| Pool 3 | AdaptiveAvgPool2d(8×8) | 128×8×8 |
| Flatten | Flatten | 8192 |
| FC1 | Linear(8192→64) + ReLU | 64 |
| Dropout | Dropout(0.5) | 64 |
| 输出 | Linear(64→10) | 10 |

---

## 📁 项目文件说明

| 文件名 | 说明 |
| :--- | :--- |
| `cifar10_CNN.py` | 模型定义与训练脚本。**注意**：数据集路径需在代码中自行修改为本地 CIFAR-10 存放位置（代码中已标记为 `你的地址`）。 |
| `cifar10_best_model.pth` | 训练50轮后保存的最佳模型权重，可直接用于预测。 |
| `test_byusingmodel.py` | 基于 Gradio 的网页预测程序。支持上传图片或拍照识别，输出预测类别、置信度及所有类别的概率分布。 |

---

## 🛠️ 环境依赖

- Python 3.8+
- PyTorch
- torchvision
- gradio
- Pillow

安装依赖：

```bash
pip install torch torchvision gradio pillow
```

## 🚀 使用指南

1️⃣ 运行网页预测程序

```bash
python test_byusingmodel.py
```

启动后 Gradio 服务将在本地运行：

地址：http://127.0.0.1:7860

支持通过上传按钮选择图片，或直接使用摄像头拍照识别

预测结果将显示：

预测类别（例如：cat、dog 等）

置信度（百分比）

10个类别中概率最大类别的值（概率值）

2️⃣ 重新训练模型（可选）
注意：训练脚本 cifar10_CNN.py 中的数据集路径已设为占位符 "你的地址"，请先将其替换为本机 CIFAR-10 数据集的根目录路径。
同时可自行调整其中关于迭代训练过程中的参数

```bash
python cifar10_CNN.py
```

训练过程中会在验证集上评估性能，并保存最佳模型为 cifar10_best_model.pth

📌 注意事项
数据集路径：本项目未包含 CIFAR-10 数据集文件，训练前请确保本地已存在数据集，或在代码中填写正确的下载/存储路径。

模型权重：cifar10_best_model.pth 与训练脚本中的模型结构严格对应，若修改网络结构需重新训练并生成新的权重文件。

Gradio 端口：默认使用 127.0.0.1:7860，如需更改请在 test_byusingmodel.py 中调整 server_name 和 server_port 参数。

🖼️ 效果预览
Gradio 网页运行时的截图，展示上传图片及预测结果。

![图片分类识别程序运行截图](images/demo_car.png)

