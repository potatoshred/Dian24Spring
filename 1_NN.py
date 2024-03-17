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
logging.basicConfig(level=LOGLEVEL, format='%(asctime)s[%(levelname)s]: %(message)s')
Info = logging.info
Warn = logging.warn


current_dir = os.path.dirname(os.path.abspath(__file__))
mnist_dir = os.path.join(current_dir, "MNIST/raw")

# 解压缩
_, _, mnist_files = next(os.walk(mnist_dir))
for file in mnist_files:
    # 检查是否为gz压缩文件，并检查是否已解压
    if file.endswith(".gz") and file[:-3] not in mnist_files:
        input_file  = os.path.join(mnist_dir, file)
        output_file = input_file.rsplit('.')[0]
        Info(f"unzipping {input_file}")
        with gzip.open(input_file, 'rb') as fi:
            with open(output_file, 'wb') as fo:
                shutil.copyfileobj(fi, fo)


# 构建全连接网络
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载本地MNIST数据集
train_dataset = datasets.MNIST(root=current_dir, train=True, download=False, transform=transforms.ToTensor())  # 必须放在当前目录的/MINST/raw目录下
test_dataset  = datasets.MNIST(root=current_dir, train=False, download=False, transform=transforms.ToTensor())
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=64, shuffle=False)


net = FCNN()
loss = nn.CrossEntropyLoss()
trainer = optim.Adam(net.parameters(), lr=0.001)


# 训练
def Train(model, loss, optimizer, train_loader, epochs=5):
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
        Info(f"Epoch {epoch+1}/{epochs}, Loss: {l.item()}")


# 测试模型
def Test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    confusion_mat = np.zeros((10, 10))
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for row, col in zip(labels, predicted):
                confusion_mat[row, col] += 1

    # 评测指标：采用加权平均
    diag      = confusion_mat.diagonal()
    weights   = confusion_mat.sum(axis=0)/confusion_mat.sum()
    Accuracy  = np.multiply(weights, diag/confusion_mat.sum()).sum()
    Precision = np.multiply(weights, diag/confusion_mat.sum(axis=0)).sum()
    Recall    = np.multiply(weights, diag/confusion_mat.sum(axis=1)).sum()
    F1_score  = 2*Precision*Recall/(Precision+Recall)
    Info("---Metrics---")
    Info(f"{'Accuracy':10s}:{Accuracy}")
    Info(f"{'Precision':10s}:{Precision}")
    Info(f"{'Recall':10s}:{Recall}")
    Info(f"{'F1_score':10s}:{F1_score}")


# 运行训练和测试
Info("---Start training---")
Train(net, loss, trainer, train_loader, epochs=3)
Test(net, test_loader)
