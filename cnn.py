# ####################
# 这段代码默认为在装有cuda的服务器上运行。
# 这段代码用于预处理所需的数据，然后训练和测试CNN网络。可以选择导入一个旧的模型继续训练、或训练一个新的模型。
# 可以调整学习率、batch大小、分类阈值、训练与测试集的分割比例、epoch数等参数。
# 命令行参数请参考argparse部分。
# ####################
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split, Subset, ConcatDataset, SubsetRandomSampler
from torchvision.transforms import Lambda
from torch import Tensor
import numpy as np
import json
import argparse
import os
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-w', nargs='*', help="white sample directories", default=["/home/ubuntu/data/sandbox/80+", "/home/ubuntu/data/sandbox/crawl_white"])
parser.add_argument('-m', nargs='*', help="black sample directories", default=["/home/ubuntu/data/sandbox/report"])
parser.add_argument('-o', type=str, help="out path of preprocessed data", default="/home/ubuntu/data/sandbox/scripts")

# 是否进行预处理
parser.add_argument('--do-preprocess', dest='do_preprocess', action='store_true')
parser.add_argument('--no-preprocess', dest='do_preprocess', action='store_false')
parser.set_defaults(do_preprocess=True)

# 是否导入旧模型
parser.add_argument('--new', dest='new_model', action='store_true')
parser.add_argument('--old', dest='new_model', action='store_false')
parser.set_defaults(new_model=True)

# 在cuda或cpu上运行
parser.add_argument('--cuda', dest='run_on_cuda', action='store_true')
parser.add_argument('--cpu', dest='run_on_cuda', action='store_false')
parser.set_defaults(run_on_cuda=True)

# 1：训练和测试；2：仅测试；3：为了快速验证代码的正确性，从样本中抽取一些样本（默认为10）测试神经网络是否能够正常运算。
parser.add_argument('-f', type=int, help="1: training and verifying; 2: only verifying; 3: test mode", default=1)
parser.add_argument('-a', type=int, help="test mode, number of samples to test", default=10)

parser.add_argument('-b', type=int, help="batch size", default=32)
parser.add_argument('-l', type=float, help="learning rate", default=0.001)
parser.add_argument('-i', type=str, help="load model from file", default="/home/ubuntu/data/sandbox/models/cnn_20channel_fc_100_2.pth")
parser.add_argument('-p', type=str, help="save model in file", default="/home/ubuntu/data/sandbox/models/cnn_20channel_fc_100_2.pth")
parser.add_argument('-d', type=str, help="data path", default="/home/ubuntu/data/sandbox/scripts")
parser.add_argument('-t', type=float, help="threshold", default=0.5)
parser.add_argument('-q', type=float, help="split data into training and testing sets", default=0.8)
parser.add_argument('-e', type=int, help="number of epochs", default=10)
args = parser.parse_args()

KERNAL_WIDTH = 10

if args.f == 1:
    MODE = "normal"
elif args.f == 2:
    MODE = "verifying"
else:
    MODE = "test_mode"
run_on_cuda = args.run_on_cuda
threshold = args.t
data_split = args.q
save_path = args.p   
EPOCHS = args.e
test_samples = args.t
load_path = args.i
LEARNING_RATE = args.l
BATCH_SIZE=args.b
new_model = args.new_model
white_sample_dir = args.w
black_sample_dir = args.m
data_out_path = args.o
data_path = args.d
do_preprocess = args.do_preprocess

print(f"*create a new model: {new_model}")
print(f"*run on cuda: {run_on_cuda}")
print(f"*mode: {MODE}")
print(f"*epochs: {EPOCHS}")
print(f"*batch size: {BATCH_SIZE}")
print(f"*learning rate: {LEARNING_RATE}")
print(f"*loading model from: {load_path}")
print(f"*save model at: {save_path}")
print(f"*loading data from: {data_path}")

# ####################
#这个函数用于去除api名称中的"w", "A", "Ex", "ExW", "ExA"后缀
# ####################
def strip_suffixes(s:str)->str:
    if s.endswith("W") or s.endswith("A"):
        s = s[:-1]
    if s.endswith("Ex"):
        s = s[:-2]
    return s

# ####################
# 数据预处理。可以指定黑、白样本路径（可以是多个黑、白样本路径），以及预处理数据的输出路径。以上参数都有默认值。
# 预处理输出文件：
# api_to_index.json：存储api名到api对应的index的映射。index范围为1~总api调用种类
# file_api_list.json：存储每个文件的api调用序列，格式如下
# {
#     'file_name':{
#         'api_list':[]
#         'label':0 or 1
#     },
# ...
# }
# ####################
def preprocess(white_sample_dir, black_sample_dir, data_out_path):
    path_list = []
    for dir in white_sample_dir:
        path_list.append((dir, 0))
    for dir in black_sample_dir:
        path_list.append((dir, 1))
    api_dict = {}
    index = 0
    files = []
    for dir_label in path_list:
        files.extend([(dir_label[0] + '/' + dir, dir_label[1]) for dir in os.listdir(dir_label[0])])
    # label==1：黑样本; label==0：白样本

    file_list = {}
    i = 0
    for file in files:
        api_list = []
        if (i % 1000) == 0:
            print(i)
        f = open(file[0], 'rb')
        s = json.load(f)
        if "data" in s:
            s = s["data"]
        for process in s["behavior"]["processes"]:
            if process["calls"] == None:
                continue
            for call in process["calls"]:
                api = call["api"]
                api = strip_suffixes(api)
                if api not in api_dict:
                    index += 1
                    api_dict[api] = index
                api_list.append(api_dict[api])
        if len(api_list):
            file_list[file[0].split('/')[-1]] = {"api_list": api_list, "label":file[1]}
        i += 1
    with open(f"{data_out_path}/api_to_index.json", 'w') as f:
        json.dump(api_dict, f, indent=1)
    with open(f"{data_out_path}/file_api_list.json", 'w') as f:
        json.dump(file_list, f)
    print("preprocessing finished")

class CustomDataset(Dataset):

    def __init__(self, file, transform=None, target_transform=None) -> None:
        raw_data = json.load(open(file, 'rb'))
        self.labels = [raw_data[key]['label'] for key in raw_data]
        self.data = [raw_data[key]['api_list'] for key in raw_data]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

# ####################
# 自定义collate_fn：由于每个函数的api调用序列长度不同。需要将每个batch中所有样本的长度补至相同。
# sample_list变量的格式：
# [([Tensor([feature vector of a api]), ...], label), ...]
# ####################
def CustomCollate(sample_list):
    # 一些样本api调用次数比卷积核长度小，因此至少需要补至卷积核的长度
    max_len = max([len(sample[0]) for sample in sample_list] + [KERNAL_WIDTH])
    labels = []
    padded_sample_list = []
    # 将一个batch中所有样本的长度补至相同
    for sample in sample_list:
        padded_sample_list.append(torch.cat((torch.stack(sample[0]), torch.zeros([max_len - len(sample[0]), len(sample[0][0])]))))
        labels.append(sample[1])
    return torch.stack(padded_sample_list).unsqueeze(1), torch.LongTensor(labels)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 20, (KERNAL_WIDTH, 343)),#[batch_size, channel, length - kernel_length + 1, 1]
            nn.ReLU(),
            nn.Flatten(2),#[batch_size, channel, length - kernel_length + 1]
            # 这步最大池化有待商榷
            nn.AdaptiveMaxPool1d(100),#[batch_size, channel, 100]
            nn.Flatten(1),#[batch_size, channel * 100]
            nn.Linear(2000, 100),#[batch_size, 100]
            nn.ReLU(),
            nn.Linear(100, 2),#[batch_size, 2]
        )

    def forward(self, x):
        logits = self.cnn(x)
        return logits

#训练
def train_loop(dataloader, model, loss_fn, optimizer, run_on_cuda:bool):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        
        # to run on server's GPU
        if run_on_cuda:
            X = X.cuda()
            y = y.cuda()

        pred = model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#测试
def test_loop(dataloader, model, loss_fn, threshold, run_on_cuda:bool):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        #混淆矩阵
        c_m = np.zeros((2,2))

        for X, y in dataloader:
            if run_on_cuda:
                X = X.cuda()
                y = y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #修改阈值
            pred = nn.functional.softmax(pred, dim=1)
            pred = nn.functional.threshold(pred, threshold, 0)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_pred = pred.argmax(1)
            for i in range(len(y)):
                c_m[y[i], y_pred[i]] += 1

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    #打印混淆矩阵，计算检出率和误报率
    print(f"confusion matrix:{c_m}")
    print(f"detected rate:{c_m[1][1] / (c_m[1][0] + c_m[1][1]) * 100:>0.1f}%")
    print(f"false positive rate:{c_m[0][1] / (c_m[0][0] + c_m[0][1]) * 100:>0.1f}%")
    print(f'sum of confusion matrix:{c_m[0][0] + c_m[0][1] + c_m[1][0] + c_m[1][1]}')

if do_preprocess:
    preprocess(white_sample_dir, black_sample_dir, data_out_path)
with open(f"{data_path}/api_to_index.json", 'r') as f:
    api_index_dic = json.load(open(f"{data_path}/api_to_index.json", 'r'))
# feature_width：每个api对应的特征向量的长度。由于将api对应的index从整数转化为独热码，feature_width就是不同api的总量。
feature_width = max([api_index_dic[key] for key in api_index_dic])

if (MODE == "verifying"):
    #使用lambda函数将api对应的index从整数转化为独热码。
    test_data = CustomDataset(f"{data_path}/file_api_list.json",
                            transform=Lambda(lambda y: [torch.zeros(feature_width, dtype=torch.float).scatter_(dim=0, index=torch.tensor(index - 1), value=1) 
                            for index in y]))
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=CustomCollate)
else:
    dataset = CustomDataset(f"{data_path}/file_api_list.json",
                            transform=Lambda(lambda y: [torch.zeros(feature_width, dtype=torch.float).scatter_(dim=0, index=torch.tensor(index - 1), value=1) 
                            for index in y]))

    # 平衡黑白样本，让它们的数量相同
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
        lengths=[int(len(dataset) * data_split), len(dataset) - int(len(dataset) * data_split)]
    )

    if MODE == "test_mode":
        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=CustomCollate, sampler=SubsetRandomSampler([i for i in range(test_samples)], generator=None))
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=CustomCollate, sampler=SubsetRandomSampler([i for i in range(int(test_samples / 4))], generator=None))
    else:
        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=CustomCollate)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=CustomCollate)

if new_model:
    model = NeuralNetwork()
else:
    model = torch.load(load_path)
    
if run_on_cuda:
    model = model.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    if (MODE != "verifying"):
        train_loop(train_dataloader, model, loss_fn, optimizer, run_on_cuda)
    test_loop(test_dataloader, model, loss_fn, threshold, run_on_cuda)
    if MODE != "test_mode":
        torch.save(model, save_path)
print("Done!")