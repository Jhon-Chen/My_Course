"""
准备数据库
"""


from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
import conf
from sklearn.preprocessing import StandardScaler


# 自定义数据集类
class myDataset(Dataset):

    def __init__(self, train=True):
        data_path_head = r"C:\\Users\\Administrator\\Git\\CFturb\\Main_Dimension_Cal\\dataset"
        data_path_tail = r"\\01_train.csv" if train else r"\\01_test.csv"
        data = pd.read_csv(data_path_head + data_path_tail)
        std = StandardScaler()
        data = std.fit_transform(data)
        self.csv_data = data

    def __getitem__(self, idx):
        data = self.csv_data[idx, :]
        q = data[1]
        h = data[2]
        n = data[3]
        npsh = data[4]
        beta2 = data[9]
        d0k0 = data[10]
        d2k2d2 = data[11]
        b2k2b2 = data[12]
        return q, h, n, npsh, beta2, d0k0, d2k2d2, b2k2b2

    def __len__(self):
        return len(self.csv_data)


def get_dataloader(train=True):
    dataset = myDataset(train)
    return DataLoader(dataset, batch_size=1, shuffle=True)


if __name__ == '__main__':

    myDataset = myDataset(train=True)
    dataloader = DataLoader(dataset=myDataset, batch_size=1)
    for idx, (q, h, n, npsh, beta2, d0k0, d2k2d2, b2k2b2) in enumerate(dataloader):
        print(idx)
        print(q.shape)
        print(h.dtype)
        print(n.dtype)
        print(npsh)
        print(q)
        print(h)
        print(n)
        print(d0k0)
        # q = q.float()
        # h = h.float()
        # n = n.float()
        # x = torch.cat((q, h, n), 0)
        # print(x.dtype)
        # print(beta2)
        # print(d0k0)
        # print(d2k2d2)
        # print(b2k2b2)
        # print(type(q), type(h), type(n))
        # df = pd.DataFrame(list(zip(q, h, n)))
        # print(df)
        break
    print(len(dataloader))

    # print(dataset.iloc[:, 10:].iloc[:, :1], dataset.iloc[:, 10:].iloc[:, 2:])
    # print(myDataset.info())
    # a = dataset.iloc[:, 10:].iloc[:, :1]
    # b = dataset.iloc[:, 10:].iloc[:, 2:]
    # c = pd.concat([a, b], axis=1)
    # print(c)

