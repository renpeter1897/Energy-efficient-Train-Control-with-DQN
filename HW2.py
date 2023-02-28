import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import gc

configs = {
    'concat_nframes': 13,
    'train_ratio': 0.9,
    'seed': 0,
    'batch_size': 200,
    'num_epoch': 25,
    'lr': 0.0001,
    'model_path': r'hw2/model.pth',
    'input_dim': 39 * 13,
    'hidden_layers': 2,
    'hidden_dim': 1700
}


# %%  数据预处理部分，从原始波形中提取MFCC特征（看求不懂，也没注释，改日再说吧）
def load_feat(path):
    feat = torch.load(path)
    return feat


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        # repeat函数: repeat(a, b)，将对应维度的tensor复制a次
        # 如: 第0维复制a次，第1维复制b次，参数数量与tensor维度不相等时也可使用
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x
    return torch.cat((left, right), dim=0)


def concat_feat(x, concat_n):
    assert concat_n % 2 == 1
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)
    mid = (concat_n // 2)
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio, train_val_seed=1337):
    class_num = 41  # NOTE: pre-computed, should not need change
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
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(
        len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
            label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
            y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
        y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
        print(y.shape)
        return X, y
    else:
        return X


# %% 定义Dataset和网络架构
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


class BasicBlock(nn.Module):
    # 网络中的基础模块，input经过一层线性整合以后用Relu激活输出
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            # nn.Sequential():允许将多个网络结构封装为一个模块，使用该方法能实现自己定义网络的不同layer
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(p=0.35)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_dim):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            # *号表示动态形参，预先并不知道有多少个参数传入函数
            # 取决于hidden_layers的数目，有多少个hidden_layers，BasicBlock在这里就执行多少次
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# %% train & valid
train_X, train_Y = preprocess_data(split='train',
                                   feat_dir=r'hw2/libriphone/libriphone/feat',
                                   phone_path=r'hw2/libriphone/libriphone',
                                   concat_nframes=configs['concat_nframes'],
                                   train_ratio=configs['train_ratio'])

val_X, val_Y = preprocess_data(split='val',
                               feat_dir=r'hw2/libriphone/libriphone/feat',
                               phone_path=r'hw2/libriphone/libriphone',
                               concat_nframes=configs['concat_nframes'],
                               train_ratio=configs['train_ratio'])

train_set = LibriDataset(train_X, train_Y)
val_set = LibriDataset(val_X, val_Y)

del train_X, train_Y, val_X, val_Y
gc.collect()

train_loader = DataLoader(train_set,
                          batch_size=configs['batch_size'],
                          shuffle=True)
val_loader = DataLoader(val_set,
                        batch_size=configs['batch_size'],
                        shuffle=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(configs['seed'])

model = Classifier(input_dim=configs['input_dim'],
                   hidden_layers=configs['hidden_layers'],
                   hidden_dim=configs['hidden_dim'],
                   output_dim=41
                   ).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=configs['lr'])

best_acc = 0.0
writer = SummaryWriter('hw2_logs')
for epoch in range(configs['num_epoch']):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, train_pred = torch.max(outputs, 1)
        # 获得最大概率分类的下标,_返回概率，train_pred返回下标
        train_acc += (train_pred.detach() == labels.detach()).sum().detach().cpu().item()
        train_loss += loss.detach().cpu().item()

    if len(val_set) > 0:
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)

                loss = criterion(outputs, labels)

                _, val_pred = torch.max(outputs, 1)
                val_acc += (val_pred.cpu() == labels.cpu()).sum().detach().cpu().item()
                val_loss += loss.detach().cpu().item()
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, configs['num_epoch'], train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), configs['model_path'])
                print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, configs['num_epoch'], train_acc / len(train_set), train_loss / len(train_loader)
        ))
    train_acc = train_acc / len(train_set)
    writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=epoch + 1)
    writer.add_scalar(tag='train_acc', scalar_value=train_acc, global_step=epoch + 1)
    val_acc = val_acc / len(val_set)
    writer.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=epoch + 1)
    writer.add_scalar(tag='val_acc', scalar_value=val_acc, global_step=epoch + 1)

if len(val_set) == 0:
    torch.save(model.state_dict(), configs['model_path'])
    print('saving model at last epoch')

# %% test
test_X = preprocess_data(split='test',
                         feat_dir=r'hw2/libriphone/libriphone/feat',
                         phone_path=r'hw2/libriphone/libriphone',
                         concat_nframes=configs['concat_nframes'],
                         train_ratio=configs['train_ratio'])
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set,
                         batch_size=configs['batch_size'],
                         shuffle=False)

test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)
model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1)
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

with open(r'hw2/prediction.csv', 'w') as f:
    f.write('Id, Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))
