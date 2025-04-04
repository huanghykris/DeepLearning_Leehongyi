import os
import random
import pandas as pd
import torch
from tqdm import tqdm

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x,n):
    if n < 0:
        left = x[0].repeat(-n,1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n,1)
        left = x[:n]
    else:
        return x

    return torch.cat((left,right),dim = 0)

# 由于一个音素（phoneme）可能跨越多个帧，并且其识别依赖于上下文帧（过去和未来的帧），因此我们需要将相邻的音素帧拼接起来进行训练，以提高识别准确率。
# concat_feat 函数的作用是：
# 将当前帧及其相邻的 k 个过去帧和 k 个未来帧（共 2k+1 = n 帧）拼接成一个特征向量。
# 最终预测的是中心帧（即当前帧）的音素类别。
# 关键要求
# 不要丢弃任何帧（即使修改预处理函数，也要确保总帧数与课件中提到的保持一致）。
# 如果修改预处理逻辑，请检查帧数是否正确匹配。

def concat_feat(x,concat_n):
    """

    :param x: 形状为 (seq_len, feature_dim) 的张量，表示一个语音序列的 MFCC 特征（seq_len 是帧数，feature_dim 是每帧的特征维度）。
    :param concat_n: concat_n：要拼接的帧总数（必须是奇数，如 3, 5, 7,...，确保中心帧存在）。
    :return: 形状为 (seq_len, concat_n * feature_dim) 的张量，表示每一帧与其相邻 (concat_n // 2) 帧拼接后的特征。
    """
    assert concat_n % 2 == 1  # 确保 concat_n 是奇数（如 3, 5, 7,...）
    if concat_n < 2:  # 若 concat_n=1，直接返回原特征
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)  # 形状变为 (seq_len, concat_n * feature_dim)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)  # 变形为 (concat_n, seq_len, feature_dim)
    mid = (concat_n // 2)  # 中心帧的索引（如 concat_n=3 时，mid=1）
    for r_idx in range(1, mid + 1):
        # 向右移位：处理中心帧右侧的帧（未来帧）
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        # 向左移位：处理中心帧左侧的帧（过去帧）
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_path, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    """

    :param split:数据集划分类型（'train', 'val', 'test'）。
    :param feat_dir:存放 MFCC 特征的目录。
    :param phone_path:存放音素标签和划分信息的目录。
    :param concat_nframes:拼接的帧数（必须是奇数，如 3, 5, 7,...）。
    :param train_ratio:训练集比例（默认 0.8）。
    :param train_val_seed:随机种子，用于可重复划分训练集和验证集（默认 1337）。
    :return:
    """
    class_num = 41  # 音素类别数（固定为41类）
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
        phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()
        for line in phone_file:
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances  for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000  # 预分配足够大的空间
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i,fname in tqdm(enumerate(usage_list), total=len(usage_list)):
        feat = load_feat(os.path.join(feat_path,mode,f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
            label = torch.LongTensor(label_dict[fname])

        X[idx:idx+cur_len,:] = feat
        if mode != 'test':
            y[idx:idx + cur_len] = label

        idx += cur_len

    X = X[:idx,:]
    if mode != 'test':
        y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
        print(y.shape)
        return X, y
    else:
        return X

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Classifier(nn.Module):
    def __init__(self,input_dim,output_dim=41,hidden_layers=1,hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim,output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# Hyper-parameters
# data parameters
concat_nframes = 1              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.8               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 0                        # random seed
batch_size = 512                # batch size
num_epoch = 5                   # the number of training epoch
learning_rate = 0.0001          # learning rate
model_path = '.\\model.ckpt'     # the path where the checkpoint will be saved

# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 1               # the number of hidden layers
hidden_dim = 256                # the hidden dim

import gc

train_X,train_y = preprocess_data(split='train',feat_path='.\\libriphone\\feat',phone_path='.\\libriphone',
                                  concat_nframes=concat_nframes,train_ratio=train_ratio)
val_X,val_y = preprocess_data(split='val',feat_path='.\\libriphone\\feat',phone_path='.\\libriphone',
                              concat_nframes=concat_nframes,train_ratio=train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y,val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

import numpy as np

# fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(seed)

# create model
model = Classifier(input_dim=input_dim,hidden_layers=hidden_layers,hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()
    for i,batch in enumerate(tqdm(train_loader)):
        features,labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _,train_pred = torch.max(outputs, 1)
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()

    # validation
    if len(val_set) > 0:
        model.eval()
        with torch.no_grad():
            for i,batch in enumerate(tqdm(val_loader)):
                features,labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)

                loss = criterion(outputs, labels)

                _,val_pred = torch.max(outputs, 1)
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))

            # if model improves,save a checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))
# if not validation save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))

del train_loader, val_loader
gc.collect()

# test
test_X = preprocess_data(split='test',feat_path='.\\libriphone\\feat',phone_path='.\\libriphone',concat_nframes=concat_nframes)
test_set = LibriDataset(test_X,None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = Classifier(input_dim=input_dim,hidden_layers=hidden_layers,hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))

test_acc = 0.0
test_lengths = 0
pred = np.array([],dtype=np.int32)

model.eval()
with torch.no_grad():
    for i,batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)
        outputs = model(features)

        _,test_pred = torch.max(outputs, 1)
        pred = np.concatenate((pred, test_pred.cpu().numpy()),axis=0)
        # NumPy 只能处理 CPU 上的数据，无法直接访问 GPU 张量。如果 test_pred 在 GPU 上，直接调用 .numpy() 会报错：

with open('prediction.csv','w') as f:
    f.write('Id,Class\n')
    for i,y in enumerate(pred):
        f.write('{},{}\n'.format(i,y))



















