# 说话人识别系统

## 项目概述

本项目实现了一个基于深度学习的说话人识别系统，能够从语音片段中识别说话人身份。系统使用梅尔频谱特征作为输入，通过Transformer编码器提取特征，最终分类到预定义的说话人类别。

## 主要功能

- 语音特征数据加载与预处理
- 梅尔频谱特征分段处理
- Transformer编码器特征提取
- 说话人分类模型训练
- 模型验证与测试
- 结果导出

## 环境要求

- Python 3.x
- PyTorch 1.8+
- NumPy
- tqdm
- CUDA (推荐)

## 文件结构

```
./
├── Dataset/
│   ├── mapping.json          # 说话人ID映射文件
│   ├── metadata.json         # 训练数据元数据
│   ├── testdata.json         # 测试数据元数据
│   └── features/             # 特征文件目录
├── model.ckpt                # 训练好的模型权重
└── output.csv                # 预测结果文件
```

## 快速开始

1. 安装依赖包：
```bash
pip install torch numpy tqdm
```

2. 准备数据：
- 将语音特征数据放入`Dataset`目录下
- 确保包含`mapping.json`、`metadata.json`和`testdata.json`

3. 训练模型：
```python
python speaker_recognition.py --mode train
```

4. 运行推理：
```python
python speaker_recognition.py --mode inference
```

## 配置参数

### 训练配置
```python
{
    "data_dir": "./Dataset",
    "save_path": "model.ckpt",
    "batch_size": 32,
    "n_workers": 8,
    "valid_steps": 2000,
    "warmup_steps": 1000,
    "save_steps": 10000,
    "total_steps": 70000
}
```

### 推理配置
```python
{
    "data_dir": "./Dataset",
    "model_path": "./model.ckpt",
    "output_path": "./output.csv"
}
```

## 数据预处理

### 梅尔频谱特征处理
- 从语音中提取40维梅尔频谱特征
- 将特征切分为128帧的片段
- 对短于128帧的特征进行零填充

### 数据集划分
- 训练集: 90%
- 验证集: 10%

## 模型架构

### 特征提取
```python
self.prenet = nn.Linear(40, d_model)  # 40维梅尔频谱特征投影
```

### Transformer编码器
```python
self.encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    dim_feedforward=256,
    nhead=2
)
self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
```

### 分类器
```python
self.pred_layer = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.ReLU(),
    nn.Linear(d_model, n_spks),
)
```

## 训练策略

- 优化器: AdamW
- 学习率调度: 余弦退火 + 热身
- 损失函数: 交叉熵损失
- 批量大小: 32
- 总训练步数: 70,000

## 结果输出

预测结果保存为CSV文件，格式如下：

```
Id,Category
path/to/feature1.pt,speaker1
path/to/feature2.pt,speaker2
...
```

## 注意事项

1. 确保数据路径正确
2. 根据GPU显存调整batch_size
3. 可修改total_steps增加训练步数
4. 模型架构可在Classifier类中修改
5. 推荐使用GPU进行训练