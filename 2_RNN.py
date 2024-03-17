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


current_dir = os.path.dirname(os.path.abspath(__file__))
mnist_dir = os.path.join(current_dir, "FashionMNIST/raw")

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

# 构建RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    

# 加载本地MNIST数据集
train_dataset = datasets.FashionMNIST(
    root=current_dir, train=True, download=False, transform=transforms.ToTensor())  # 必须放在当前目录的/MINST/raw目录下
test_dataset = datasets.FashionMNIST(
    root=current_dir, train=False, download=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Parameter
input_size = 28  # 每一行像素作为一个时间步
hidden_size = 128
num_layers = 2
num_classes = 10
sequence_length = 28  # 序列长度，即图像高度

net = RNN(input_size, hidden_size, num_layers, num_classes)
loss = nn.CrossEntropyLoss()
trainer = optim.Adam(net.parameters(), lr=0.0004)

# 训练模型
epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        l = loss(outputs, labels)
        
        trainer.zero_grad()
        l.backward()
        trainer.step()
        
        if (i+1) % 100 == 0:
            Info(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {l.item()}')

# 在测试集上评估模型
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    Info(f'Accuracy of the model on the test images: {(100 * correct / total)}%')


# # 基本评测指标
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# inputs, labels = next(iter(test_loader))
# model = net
# model.eval()
# outputs = model(inputs)
# _, predicted = torch.max(outputs.data, 1)

# # 计算混淆矩阵
# confusion_mat = np.zeros((10, 10))
# for row, col in zip(labels, predicted):
#     confusion_mat[row, col] += 1

# diag = confusion_mat.diagonal()

# # 采用加权平均
# weights = confusion_mat.sum(axis=0)/confusion_mat.sum()
# Accuracy = diag.sum()/confusion_mat.sum()
# Precision = np.multiply(weights, diag/confusion_mat.sum(axis=0)).sum()
# Recall = np.multiply(weights, diag/confusion_mat.sum(axis=1)).sum()
# F1_score = 2*Precision*Recall/(Precision+Recall)

# Info("---Metrics---")
# Info(f"{'Accuracy':10s}:{Accuracy}")
# Info(f"{'Precision':10s}:{Precision}")
# Info(f"{'Recall':10s}:{Recall}")
# Info(f"{'F1_score':10s}:{F1_score}")