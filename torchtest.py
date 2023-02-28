import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.feature_selection import SelectKBest
import torch
from sklearn.feature_selection import f_regression
from torch import optim
import torch.nn as nn
import pandas as pd
import csv


def get_feature_importance(feature_data, label_data, k=4, column=None):
    selector = SelectKBest(score_func=f_regression, k=k)
    res = selector.fit(feature_data, label_data)
    idx = np.argsort(res.scores_)[::-1]  # 返回得分从大到小的特征的索引值
    x_new = feature_data.iloc[:, idx[:k]]
    print(f'Top {k} Best feature score ')
    print(res.scores_[idx[:k]])
    print(f'\nTop {k} Best feature name')
    print(feature_data.columns[idx[:k]])
    return x_new, idx


class covidDataset(Dataset):
    def __init__(self, path, mode, k):
        df = pd.read_csv(path)
        column = df.columns
        train_x, train_y = df.iloc[:, 41:-1], df.iloc[:, -1]
        x_new, col_indices = get_feature_importance(train_x, train_y, k, column)
        x_new = np.array(x_new).astype(float)
        y_new = np.array(train_y).astype(float)
        if mode == 'train':  # 如果读的是训练数据 就逢5取4  indices是数据下标
            indices = [i for i in range(len(x_new)) if i % 5 != 0]
            self.y = torch.tensor(y_new[indices])  # 取标签
        elif mode == 'val':  # 如果读的是验证数据 就逢5取1  indices是数据下标
            indices = [i for i in range(len(x_new)) if i % 5 == 0]
            self.y = torch.tensor(y_new[indices])  # 取标签
        else:  # 如果读的是测试数据 就全取了
            indices = [i for i in range(len(x_new))]
        data = torch.tensor(x_new[indices])  # 取数据
        self.data = data  # 取数据
        self.mode = mode
        self.data = (self.data - self.data.mean(dim=0, keepdim=True)) / self.data.std(dim=0,
                                                                                      keepdim=True)  # 这里将数据归一化。
        # assert k == self.data.shape[1]
        # print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
        #       .format(mode, len(self.data), k))

    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.data[idx].float()
        else:
            return self.data[idx].float(), self.y[idx].float()

    def __len__(self):
        return len(self.data)


class myNet(nn.Module):
    def __init__(self, inDim):
        super(myNet, self).__init__()
        self.fc1 = nn.Linear(inDim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if len(x.size()) > 1:
            return x.squeeze(1)
        else:
            return x


def train_val(model, trainloader, valloader, optimizer, loss, epoch, device, save_):
    model = model.to(device)  # 将模型加载到指定设备（GPU或是CPU）
    min_val_loss = 100000
    writer = SummaryWriter('logs')
    for i in range(epoch):  # 总共训练epoch次
        print(f"当前正在训练第{i}个epoch")
        # 训练集
        model.train()  # 开始训练
        train_loss = 0.0
        val_loss = 0.0
        for data in trainloader:  # 用每一个batch的data更新梯度
            optimizer.zero_grad()  # 默认梯度累加，不同的batch梯度需要从零开始计算，因此在计算每个batch开始前需要对梯度清零
            x, target = data[0].to(device), data[1].to(device)  # 从loader里面提取特征和标签
            pred = model(x)  # 预测
            bat_loss = loss(pred, target, model)  # 计算loss
            bat_loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            train_loss += bat_loss.detach().cpu().item()  # 1个epoch的loss是每一个batch的loss的加和
            # .detach()返回一个新的tensor,仍指向原变量的存放位置，不计算梯度
            # .cpu()将cuda()中的tensor拿到cpu上，不改变变量类型
            # .item()将tensor变量转换为python标量
        writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=i + 1)
        #  验证集
        model.eval()
        with torch.no_grad():  # 验证集不用计算梯度
            b = 0
            for data in valloader:
                b += 1
                val_x, val_target = data[0].to(device), data[1].to(device)
                val_pred = model(val_x)
                val_target = val_target.type(torch.FloatTensor).to(device)
                val_bat_loss = loss(val_pred, val_target, model)
                val_loss += val_bat_loss.cpu().item()
                print(f"\n第{i}个epoch的第{b}个batch的val_loss为:{val_bat_loss}")
                print(f"\n第{i}个epoch的总val_loss为:{val_loss}")
        writer.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=i + 1)
        if val_loss < min_val_loss:
            torch.save(model, save_)
            min_val_loss = val_loss  # 存储验证集loss最小的模型
        print(f"\n当前最小val_loss为:{min_val_loss}")


def evaluate(model_path, testset, rel_path, device):
    model = torch.load(model_path).to(device)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)  # 放入loader，其实可能没必要，loader作用就是把数据形成批次而已
    val_rel = []
    model.eval()
    with torch.no_grad():
        for data in testloader:
            x = data.to(device)
            pred = model(x)
            val_rel.append(pred.item())
    print(val_rel)
    with open(rel_path, 'w') as f:
        csv_writer = csv.writer(f)  # 百度的csv写法
        csv_writer.writerow(['id', 'tested_positive'])
        for i in range(len(testset)):
            csv_writer.writerow([str(i), str(val_rel[i])])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_path = r'hw1/covid.train.csv'
test_path = r'hw1/covid.test.csv'
feature_dim = 4
trainset = covidDataset(train_path, 'train', k=feature_dim)
valset = covidDataset(train_path, 'val', k=feature_dim)
testset = covidDataset(test_path, 'test', k=feature_dim)
config = {
    'n_epochs': 500,
    'batch_size': 200,
    'optimizer': 'SGD',
    'optim_hparas':
        {
            'lr': 0.001,
            'momentum': 0.9
        },
    'save_path': r'hw1/model1.pth'
}


def getLoss(pred, target, model):
    loss = nn.MSELoss(reduction='mean')
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(param ** 2)
    return loss(pred, target) + 0.00075 * regularization_loss


loss = getLoss
model = myNet(feature_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=True)
train_val(model, trainloader, valloader, optimizer, loss, config['n_epochs'], device, save_=config['save_path'])
evaluate(config['save_path'], testset, r'hw1/pred1.csv', device)