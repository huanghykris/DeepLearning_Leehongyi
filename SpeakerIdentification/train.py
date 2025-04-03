import numpy as np
import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    # 设置NumPy的随机种子，确保基于NumPy的随机操作可复现
    np.random.seed(seed)
    # 设置Python内置random模块的随机种子，确保Python层级的随机操作可复现
    random.seed(seed)
    # 设置PyTorch的CPU随机种子，确保CPU上的随机操作可复现
    torch.manual_seed(seed)
    # 检查是否可用CUDA（GPU）
    if torch.cuda.is_available():
        # 设置当前GPU的随机种子
        torch.cuda.manual_seed(seed)
        # 设置所有GPU的随机种子（适用于多GPU环境）
        torch.cuda.manual_seed_all(seed)
        # 关闭cuDNN的自动优化基准（避免不同运行间的算法选择差异）
        torch.backends.cudnn.benchmark = False
        # 启用cuDNN的确定性模式（确保卷积运算结果可复现，可能牺牲少量性能）
        torch.backends.cudnn.deterministic = True

set_seed(87)

class myDataset(Dataset):
    def __init__(self,data_dir,segment_len=128):
        self.data_dir=data_dir
        self.segment_len=segment_len

        # Loading the mapping from speaker name to their corresponding id
        mapping_path = os.path.join(self.data_dir,'mapping.json')
        mapping = json.load(open(mapping_path,'r'))
        self.speaker2id = mapping['speaker2id']

        # Load metadata of traing data
        metadata_path = os.path.join(self.data_dir,'metadata.json')
        metadata = json.load(open(metadata_path,'r'))["speakers"]

        # Get the total number of speaker
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"],self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        feature_path,speaker_id = self.data[index]
        # 加载预处理好的梅尔频谱特征
        mel = torch.load(os.path.join(self.data_dir,feature_path))

        # 将梅尔频谱切分为固定长度的片段
        if len(mel) > self.segment_len:
            # 如果特征长度大于目标长度，随机选取起始点
            start = random.randint(0,len(mel) - self.segment_len)
            # 截取从start开始的segement_len长度的片段
            mel = torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            # 如果特征长度不足，直接使用全部特征
            mel = torch.FloatTensor(mel)

        # 将说话人ID转换为LongTensor类型（用于后续的损失计算）
        speaker = torch.LongTensor([speaker_id]).long()
        # 返回梅尔频谱和对应的说话人ID
        return mel, speaker

    def get_speaker_num(self):
        return self.speaker_num

def collate_batch(batch):
    mel,speaker = zip(*batch)

    # 对变长梅尔频谱进行填充处理
    # - pad_sequence: 对变长序列按最长样本进行填充
    # - batch_first=True: 输出形状为(batch, length, feature)
    # - padding_value=-20: 填充值设为10^-20（非常小的对数域值，接近线性域的0）
    mel = pad_sequence(mel, batch_first=True, padding_value=0)
    # 输出形状：(batch_size, 最大长度, 40个梅尔频带)

    # 将说话人标签列表转换为张量，并转为long类型（用于分类任务）
    return mel,torch.FloatTensor(speaker).long()

def get_dataloader(data_dir,batch_size,n_workers):
    dataset = myDataset(data_dir)
    speaker_num = dataset.get_speaker_num()
    # 将数据集划分为训练集和验证集
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen,len(dataset) - trainlen]
    trainset,validset = random_split(dataset,lengths=lengths)

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        dataset=validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader,valid_loader,speaker_num

class Classifier(nn.Module):
    def __init__(self,d_model=80,n_spks=600,dropout=0.1):
        super().__init__()
        # 输入特征变换层：将40维梅尔频谱特征投影到d_model维度
        self.prenet = nn.Linear(40, d_model)

        # 编码器部分（原Transformer实现）
        self.encoder_layer = nn.TransformerEncoderLayer(
             d_model=d_model,
             dim_feedforward=256,
             nhead=2
         )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Confermer实现
        # from conformer import Conformer
        # self.encoder = Conformer(
        #     dim = d_model,
        #     dim_head = 64, # 注意力头维度
        #     heads = 2, # 注意力头数
        #     ff_mult = 4, #feedForward扩展倍数
        #     conv_expansion_factor = 2, # 卷积扩展因子
        #     conv_kernel_size = 31, # 卷积核大小
        #     attn_dropout = dropout,
        #     ff_dropout = dropout,
        #     conv_dropout = dropout,
        #     num_layers = 2 #层数
        # )

        # 分类预测层：
        # 1. d_model -> d_model 全连接 + ReLU
        # 2. d_model -> n_spks 全连接
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self,mels):
        """
        :param mels:(batch size,length,40)
        :return: (batch size,n_spks)
        """
        # out: (batch size,length,d_model)
        out = self.prenet(mels)
        # out: (length,batch size,d_model)
        out = out.permute(1,0,2)
        # The encoder layer expect features in the shape of (length,batch size,d_model)
        out = self.encoder_layer(out)
        # out: (batch size,length,d_model)
        out = out.transpose(0,1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch,n_spks)
        out = self.pred_layer(stats)
        return out

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    """
    创建一个学习率调度策略，该策略包含两个阶段：
    1. 热身阶段(warmup)：学习率从0线性增加到优化器设置的初始学习率
    2. 余弦衰减阶段：学习率按照余弦函数从初始值衰减到0

    参数说明：
        optimizer (:class:`~torch.optim.Optimizer`):
            需要调度学习率的优化器对象
        num_warmup_steps (:obj:`int`):
            热身阶段的步数（学习率线性增加的步数）
        num_training_steps (:obj:`int`):
            总训练步数（热身+余弦衰减的总步数）
        num_cycles (:obj:`float`, `可选`, 默认为0.5):
            余弦周期的数量（默认0.5表示半余弦周期，即从最大值降到0）
        last_epoch (:obj:`int`, `可选`, 默认为-1):
            重启训练时的最后一个epoch索引

    返回：
        :obj:`torch.optim.lr_scheduler.LambdaLR`:
            配置好的学习率调度器实例
    """
    def lr_lambda(current_step: int):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1,num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1,num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def model_fn(batch,model,criterion,device):
    mels,labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)

    loss = criterion(outs,labels)

    # Get highest probability
    preds = outs.argmax(dim=1)
    # Compute accuracy
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy

from tqdm import tqdm
def valid(dataloader,model,criterion,device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset),ncols=0,desc='Valid',unit=' uttr')

    for i,batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch,model,criterion,device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i+1):.2f}",
            accuracy=f"{running_accuracy / (i+1):.2f}"
        )

    pbar.close()
    model.train()
    return running_accuracy/len(dataloader)

# Main function
def parse_args():
    config = {
        "data_dir": ".\\Dataset",
        "save_path": "model.ckpt",
        "batch_size": 32,
        "n_workers": 8,
        "valid_steps":2000,
        "warmup_steps":1000,
        "save_steps":10000,
        "total_steps":70000,
    }
    return config

from torch.optim import AdamW
def main(
        data_dir,
        save_path,
        batch_size,
        n_workers,
        valid_steps,
        warmup_steps,
        total_steps,
        save_steps,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader,valid_loader,speaker_num = get_dataloader(data_dir,batch_size,n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish lodaing data!",flush=True)

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!",flush=True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps,ncols=0,desc='Train',unit=' step')

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss,accuracy = model_fn(batch,model,criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Update model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        #Log
        pbar.update()
        pbar.set_postfix(
            desc='Train',
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(valid_loader,model,criterion,device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps,ncols=0,desc='Valid',unit=' step')

        # Save best model so far
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict,save_path)
            pbar.write(f"Step {step + 1},best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()

# if __name__ == '__main__':
#     main(**parse_args())

from pathlib import Path
class InferenceDataset(Dataset):
    def __init__(self,data_dir):
        testdata_path = os.path.join(data_dir,'testdata.json')
        metadata = json.load(open(testdata_path))
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir,feat_path))

        return feat_path,mel

def inference_collate_batch(batch):
    feat_paths,mels = zip(*batch)
    return feat_paths,torch.stack(mels)

import csv
from tqdm import tqdm

def parse_args():
	"""arguments"""
	config = {
		"data_dir": ".\\Dataset",
		"model_path": ".\\model.ckpt",
		"output_path": ".\\output.csv",
	}

	return config

def main(
        data_dir,
        model_path,
        output_path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    mapping_path = os.path.join(data_dir,'mapping.json')
    mapping = json.load(open(mapping_path))

    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=inference_collate_batch,
    )
    print(f"[Info]: Finish loading data!",flush=True)

    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[Info]: Finish creating model!",flush=True)

    results = [["Id","Category"]]
    for feat_path,mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path,pred in zip(feat_path,preds):
                results.append([feat_path,mapping["id2speaker"][str(pred)]])

    with open(output_path,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

if __name__ == '__main__':
    main(**parse_args())









