# 语音音素识别项目

## 项目概述

本项目实现了一个基于深度学习的语音音素识别系统，能够将输入的语音MFCC特征分类为41种不同的音素类别。项目使用PyTorch框架，实现了包含上下文帧拼接的特征处理和多层感知机分类模型。

## 主要功能

- 语音特征数据加载与预处理
- 上下文帧拼接处理
- 数据集的训练集/验证集划分
- 多层感知机分类模型实现
- 模型训练与验证
- 测试集预测与结果导出

## 环境要求

- Python 3.x
- PyTorch
- NumPy
- pandas
- tqdm

## 文件结构

```
./
├── libriphone/
│   ├── feat/                # 存放MFCC特征
│   │   ├── train/           # 训练集特征
│   │   ├── val/             # 验证集特征
│   │   └── test/            # 测试集特征
│   └── labels/              # 音素标签文件
│       ├── train_labels.txt
│       ├── train_split.txt
│       └── test_split.txt
├── model.ckpt               # 训练好的模型权重
└── prediction.csv           # 预测结果文件
```

## 快速开始

1. 安装依赖包：
```bash
pip install torch numpy pandas tqdm
```

2. 准备数据：
- 将语音特征数据放入`libriphone/feat`目录下
- 确保包含`train`、`val`和`test`子文件夹
- 准备相应的标签文件

3. 运行训练：
```python
python phoneme_recognition.py
```

4. 预测结果将自动保存为`prediction.csv`

## 配置参数

主要可配置参数：

```python
# 数据参数
concat_nframes = 1       # 拼接的帧数(必须为奇数)
train_ratio = 0.8        # 训练集比例

# 训练参数
seed = 0                 # 随机种子
batch_size = 512         # 批大小
num_epoch = 5            # 训练轮数
learning_rate = 0.0001   # 学习率
model_path = './model.ckpt' # 模型保存路径

# 模型参数
input_dim = 39 * concat_nframes  # 输入维度(勿修改)
hidden_layers = 1        # 隐藏层数
hidden_dim = 256         # 隐藏层维度
```

## 数据预处理

### 上下文帧拼接
`concat_feat`函数将当前帧与其相邻帧拼接，增强上下文信息：

```python
def concat_feat(x, concat_n):
    """拼接当前帧与其相邻帧"""
    assert concat_n % 2 == 1  # 确保为奇数
    if concat_n < 2:
        return x
    # ...实现细节...
```

### 数据加载与划分
`preprocess_data`函数处理数据加载和划分：

```python
def preprocess_data(split, feat_path, phone_path, concat_nframes, train_ratio=0.8):
    """加载并预处理数据"""
    # ...实现细节...
```

## 模型架构

### 基础块
```python
class BasicBlock(nn.Module):
    """基础网络块(线性层+ReLU)"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
```

### 分类器
```python
class Classifier(nn.Module):
    """音素分类器"""
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )
```

## 训练过程

- 使用Adam优化器
- 交叉熵损失函数
- 训练集/验证集监控
- 最佳模型保存
- 训练进度条显示

## 结果输出

预测结果保存为CSV文件，格式如下：

```
Id,Class
0,12
1,25
2,7
...
```

## 注意事项

1. 确保数据路径正确
2. 根据GPU显存调整batch_size
3. 可修改num_epoch增加训练轮数
4. 可调整concat_nframes改变上下文帧数(必须为奇数)
5. 模型架构可在Classifier类中修改