# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset

# This is for the progress bar.
from tqdm.auto import tqdm
import random

_exp_name = "sample"

myseed = 42 # set a random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)

test_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

train_tfm = transforms.Compose([
    # 调整图像大小（128x128）
    transforms.Resize((128, 128)),
    # 随机水平翻转（50%概率）
    transforms.RandomHorizontalFlip(p=0.5),
    # 随机垂直翻转（20%概率）
    transforms.RandomVerticalFlip(p=0.2),
    # 随机旋转（-15° 到 15°）
    transforms.RandomRotation(15),
    # 颜色扰动（调整亮度、对比度、饱和度）
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # 随机仿射变换（平移、缩放）
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # 随机灰度化（20%概率）
    transforms.RandomGrayscale(p=0.2),
    # 高斯模糊（轻微模糊）
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    # 转换为 Tensor（必须放在最后）
    transforms.ToTensor(),

    # 标准化（可选，通常用于预训练模型）
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class FoodDataset(Dataset):
    def __init__(self,path,tfm=test_tfm,files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith('.jpg')])
        if files is not None:
            self.files = files

        # 调试：检查路径和文件列表
        # print("Debug - Path:", path)
        # print("Debug - Files:", self.files)  # 这里应该是列表，如果是None则说明路径错误
        print(f"One {path} sample",self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("\\")[-1].split("_")[0]) # 注意Windows路径用反斜杠
        except:
            label = -1 # test data has no label

        return im,label

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        # torch.nn.MaxPool2d(kernel_size,stride,padding)
        # input 维度 [3,128,128]
        self.cnn = nn.Sequential(
            # out_channels=64 决定了卷积层会使用 64个不同的卷积核，每个卷积核都会对输入进行独立的计算，生成一个 新的通道。
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), # 对输出的64个通道分别计算均值和方差，进行标准化（减均值、除方差），再通过可学习的参数（γ和β）缩放和偏移。
            nn.ReLU(),  # ReLU(x) = max(0, x)
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class Residual_NetWork(nn.Module):
    def __init__(self):
        super(Residual_NetWork, self).__init__()

        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
        )

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
        )
        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x1 = self.cnn_layer1(x)

        x1 = self.relu(x1)

        x2 = self.cnn_layer2(x1)

        x2 = self.relu(x2)

        x3 = self.cnn_layer3(x2)

        x3 = self.relu(x3)

        x4 = self.cnn_layer4(x3)

        x4 = self.relu(x4)

        x5 = self.cnn_layer5(x4)

        x5 = self.relu(x5)

        x6 = self.cnn_layer6(x5)

        x6 = self.relu(x6)

        # The extracted feature map must be flatten before going to fully-connected layers.
        xout = x6.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        xout = self.fc_layer(xout)
        return xout

batch_size = 64
_dataset_dir = ".\\data\\food11"
train_path = os.path.join(_dataset_dir, "training")
# 检查路径是否存在
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Directory {train_path} does not exist!")

# print("Debug - Path:",os.path.join(_dataset_dir,"training"))

train_set = FoodDataset(os.path.join(_dataset_dir, "training"),tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
test_set = FoodDataset(os.path.join(_dataset_dir, "validation"),tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)

# print("Deubg - cuda:", torch.cuda.is_available())
# print("Deubg - cuda:", torch.cuda.current_device())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# number of training epochs and patience
n_epochs = 4
patience = 300 # if no improvement in 'patience' epochs,early stop

#model = Classifier().to(device)
model = Residual_NetWork().to(device)
# cross-entropy
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003,weight_decay=1e-5)

# Initialize trackers,these are not parameters and should not be changed
stale = 0
best_acc = 0

def train():
    for epoch in range(n_epochs):
        # --------training--------
        model.train()

        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()
            # print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # print information
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f} acc = {train_acc:.5f}")

        # ---------- Validation ----------
        model.eval()

        valid_loss = []
        valid_accs = []

        for batch in tqdm(test_loader):
            imgs, labels = batch

            # don't need gradient in validation
            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f} acc = {valid_acc:.5f}")

        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt", "a") as f:
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt", "a") as f:
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save model
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch},saving model")
            torch.save(model.state_dict(), f"./{_exp_name}_model.ckpt")
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvement {patience} consecutive epochs,early stopping")
                break

test_set = FoodDataset(os.path.join(_dataset_dir, "validation"),tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,num_workers=0,pin_memory=True)

# model_best = Classifier().to(device)
model_best = Residual_NetWork().to(device)
model_best.load_state_dict(torch.load(f".\\{_exp_name}_model.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():  # 禁用梯度计算（节省内存）
    for data, _ in test_loader:  # 忽略标签（测试集可能无标签）
        test_pred = model_best(data.to(device))  # 前向传播
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)  # 取预测类别
        prediction += test_label.squeeze().tolist()  # 转换为列表并累积

# create test csv
def pad4(i):
    return "0"*(4-len(str(i))) + str(i)  # 补零到4位（如3→"0003"）

df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1, len(test_set)+1)]  # 生成ID列（0001,0002,...）
df["Category"] = prediction  # 写入预测类别
df.to_csv("submission.csv", index=False)  # 保存为CSV（不含行索引）

# csv文件示例
#Id,Category
#0001,3
#0002,7
#0003,1


























