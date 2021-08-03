# ####################
# 这段代码仅在本地环境（python 3.9.6）上运行过，未在服务器上运行过
# 三层全连接神经网络300+300+2
# 使用api_count_preprocess.py的数据训练该模型
# ####################
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split, Subset, ConcatDataset
from torchvision.transforms import Lambda
from torch import Tensor
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        self.data = pd.read_csv(file)
        self.labels = self.data['label']
        
        del self.data['label']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data.loc[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(343, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 2),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        c_m = np.zeros((2,2))
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            c_m += confusion_matrix(y, pred.argmax(1))

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"confusion matrix:{c_m}")

def min_max(data):
    data_max = data.max().max()
    data /= data_max
    return Tensor(data)

# ####################
# 简单的将所有黑白样本作为训练集
# 或者使用注释中的代码调整黑、白样本的比例，使它们的数量更平衡
# ####################

# 数据路径
dataset = CustomDataset("C:/Users/hhjimhhj/Desktop/实习/python_scripts/separate_count.csv", 
                        transform=min_max)

# blackset = Subset(dataset=dataset, indices=[i for i in range(len(dataset)) if dataset[i][1] == 1])
# print(len(blackset))
# whiteset = Subset(dataset=dataset, indices=[i for i in range(len(dataset)) if dataset[i][1] == 0])
# print(len(whiteset))
# blackset_shrinked, tmp= random_split(
#     dataset=blackset,
#     lengths=[3983, 19476-3983]
# )
# train_data, test_data = random_split(
#     dataset=ConcatDataset([blackset_shrinked, whiteset]),
#     lengths=[6373, 1593]
# )

train_data, test_data = random_split(
    dataset=dataset,
    lengths=[18767, 4692]
)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 使用新模型，或者导入旧模型
model = NeuralNetwork()
# model = torch.load('C:/Users/hhjimhhj/Desktop/实习/python_scripts/nn_3l_200_200.pth')

learning_rate = 1e-3

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#调整epoch数
epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

#存储模型
torch.save(model, 'C:/Users/hhjimhhj/Desktop/实习/python_scripts/nn_3l_300_300.pth')