# 食品图像分类项目

## 项目概述

本项目实现了一个基于深度学习的食品图像分类系统，能够将输入的食品图像分类到11个不同的类别中。项目使用了PyTorch框架，并实现了两种不同的神经网络架构：标准CNN和残差网络(ResNet)。

## 主要功能

- 图像数据加载与预处理
- 数据增强（训练时）
- 两种神经网络模型实现
- 模型训练与验证
- 早停机制
- 模型评估与预测
- 结果导出为CSV

## 环境要求

- Python 3.x
- PyTorch
- torchvision
- NumPy
- pandas
- PIL (Pillow)
- tqdm

## 文件结构

```
./
├── data/
│   └── food11/
│       ├── training/      # 训练集图像
│       └── validation/    # 验证集图像
├── models/                # 模型保存目录
├── sample_log.txt         # 训练日志
├── sample_model.ckpt      # 最佳模型权重
└── submission.csv         # 预测结果文件
```

## 快速开始

1. 安装依赖包：
```bash
pip install torch torchvision numpy pandas pillow tqdm
```

2. 准备数据：
- 将食品图像数据放入`data/food11`目录下
- 确保包含`training`和`validation`子文件夹

3. 运行训练：
```python
python food_classification.py
```

4. 预测结果将自动保存为`submission.csv`

## 配置参数

主要可配置参数位于代码开头：

```python
myseed = 42                # 随机种子
batch_size = 64            # 批大小
n_epochs = 4               # 训练轮数
patience = 300             # 早停耐心值
learning_rate = 0.0003     # 学习率
weight_decay = 1e-5        # 权重衰减
```

## 数据预处理

### 训练数据增强
包括随机水平/垂直翻转、旋转、颜色扰动、仿射变换、高斯模糊等：

```python
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), 
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor()
])
```

### 测试数据转换
仅包含调整大小和Tensor转换：

```python
test_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])
```

## 模型架构

### 1. 标准CNN模型

5个卷积块，每个包含：
- 卷积层
- 批归一化
- ReLU激活
- 最大池化

后接3个全连接层

### 2. 残差网络(ResNet)

6个卷积层，使用残差连接，后接2个全连接层

## 训练过程

- 使用Adam优化器
- 交叉熵损失函数
- 梯度裁剪(max_norm=10)
- 验证集监控
- 早停机制
- 最佳模型保存

## 结果输出

预测结果将保存为CSV文件，格式如下：

```
Id,Category
0001,3
0002,7
0003,1
...
```

## 注意事项

1. 确保数据路径正确
2. 根据GPU显存调整batch_size
3. 可修改n_epochs增加训练轮数
4. 可尝试不同的数据增强组合
5. 模型架构可在Classifier/Residual_Network类中修改