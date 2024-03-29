import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import gzip
import shutil

import logging
LOGLEVEL = logging.INFO
logging.basicConfig(
    level=LOGLEVEL, format='%(asctime)s[%(levelname)s]: %(message)s')
Info = logging.info
Warn = logging.warn

from tools import Visualizer, MetricsVisualizer

# Parameter
batch_size = 64
lr = 1e-3
epochs = 16

current_dir = os.path.dirname(os.path.abspath(__file__))
mnist_dir   = os.path.join(current_dir, "FashionMNIST/raw")

# 解压缩
_, _, mnist_files = next(os.walk(mnist_dir))
for file in mnist_files:
    # 检查是否为gz压缩文件，并检查是否已解压
    if file.endswith(".gz") and file[:-3] not in mnist_files:
        input_file = os.path.join(mnist_dir, file)
        output_file = input_file.rsplit('.')[0]
        Info(f"unzipping {input_file}")
        with gzip.open(input_file, 'rb') as fi:
            with open(output_file, 'wb') as fo:
                shutil.copyfileobj(fi, fo)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.U = nn.Linear(input_size, hidden_size)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        # S0
        S = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        
        # 循环计算每个时间步的输出
        for t in range(x.size(1)):
            S = torch.tanh(self.U(x[:, t, :]) + self.W(S))
        
        # 最终输出
        out = self.V(S)
        return out


# 加载本地MNIST数据集
train_dataset = datasets.FashionMNIST(root=current_dir, train=True, download=False, transform=transforms.ToTensor())  # 必须放在当前目录的/MINST/raw目录下
test_dataset  = datasets.FashionMNIST(root=current_dir, train=False, download=False, transform=transforms.ToTensor())
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Parameter
input_size      = 28*28  # 一行像素作为一个时间步
hidden_size     = 128 # 隐藏层的大小
num_classes     = 10
sequence_length = 1  # 序列长度，即图像高度


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SimpleRNN(input_size, hidden_size, num_classes).to(device)
loss = nn.CrossEntropyLoss()
trainer = optim.Adam(net.parameters(), lr=lr)
# 记录初始状态便于恢复
initial_state = net.state_dict()
initial_optimizer_state = trainer.state_dict()

# # 训练模型
# for epoch in range(epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # images = images.permute(0,1,3,2).contiguous()  # 每列像素作为一个时间步
#         images = images.view(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
        
#         outputs = net(images)
#         l = loss(outputs, labels)
        
#         trainer.zero_grad()
#         l.backward()
#         trainer.step()
        
#         if (i+1) % 100 == 0:
#             Info(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {l.item()}')

# # 在测试集上评估模型
# net.eval()
# with torch.no_grad():
#     confusion_mat = np.zeros((10, 10))
#     for images, labels in test_loader:
#         images = images.view(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         for row, col in zip(labels, predicted):
#             confusion_mat[row, col] += 1

#     # 评测指标：采用加权平均
#     diag      = confusion_mat.diagonal()
#     weights   = confusion_mat.sum(axis=1)/confusion_mat.sum()
#     Accuracy  = diag.sum()/confusion_mat.sum()
#     Precision = np.multiply(weights, diag/confusion_mat.sum(axis=0)).sum()
#     Recall    = np.multiply(weights, diag/confusion_mat.sum(axis=1)).sum()
#     F1_score  = 2*Precision*Recall/(Precision+Recall)
#     Info("---Metrics---")
#     Info(f"{'Accuracy':10s}:{Accuracy}")
#     Info(f"{'Precision':10s}:{Precision}")
#     Info(f"{'Recall':10s}:{Recall}")
#     Info(f"{'F1_score':10s}:{F1_score}")

def Test(batch_size, model):
    test_dataset  = datasets.FashionMNIST(root=current_dir, train=False, download=False, transform=transforms.ToTensor())
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    confusion_mat = np.zeros((10, 10))
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            for row, col in zip(labels, predicted):
                confusion_mat[row, col] += 1

    # 评测指标：采用加权平均
    diag      = confusion_mat.diagonal()
    weights   = confusion_mat.sum(axis=1)/confusion_mat.sum()
    Accuracy  = diag.sum()/confusion_mat.sum()
    Precision = np.multiply(weights, diag/confusion_mat.sum(axis=0)).sum()
    Recall    = np.multiply(weights, diag/confusion_mat.sum(axis=1)).sum()
    F1_score  = 2*Precision*Recall/(Precision+Recall)
    Info("---Metrics---")
    Info(f"{'Accuracy':10s}:{Accuracy}")
    Info(f"{'Precision':10s}:{Precision}")
    Info(f"{'Recall':10s}:{Recall}")
    Info(f"{'F1_score':10s}:{F1_score}")
    return Accuracy, Precision, Recall, F1_score


def Train_for_Vis(batch_size, model, loss, optimizer, epochs=5):
    # 加载本地MNIST数据集
    train_dataset = datasets.FashionMNIST(root=current_dir, train=True, download=False, transform=transforms.ToTensor())
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    out = []
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # images = images.permute(0,1,3,2).contiguous()  # 每列像素作为一个时间步
            images = images.view(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            l = loss(outputs, labels)
            
            trainer.zero_grad()
            l.backward()
            trainer.step()
            
            if (i+1) % 100 == 0:
                Info(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {l.item()}')
        out.append(Test(batch_size, model))

    return out


# Metrics Visualizer with epochs
out = Train_for_Vis(batch_size, net, loss, trainer, epochs=epochs)
MetricsVisualizer(list(zip(*out)), lr, epochs)
