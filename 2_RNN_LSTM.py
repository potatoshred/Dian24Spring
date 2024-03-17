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
train_dataset = datasets.FashionMNIST(root="./", train=True, download=False, transform=transforms.ToTensor())  # 必须放在当前目录的/MINST/raw目录下
test_dataset  = datasets.FashionMNIST(root="./", train=False, download=False, transform=transforms.ToTensor())
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=64, shuffle=False)



# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 参数设置
input_size = 28  # 每一行像素作为一个时间步
hidden_size = 128
num_layers = 2
num_classes = 10

# 实例化模型
model = LSTM(input_size, hidden_size, num_layers, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

# 训练模型
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28, input_size).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 28, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {(100 * correct / total)}%')