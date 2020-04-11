"""
准备数据库
"""


from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
import conf


# 自定义数据集类
class myDataset(Dataset):

    def __init__(self, train=True):
        data_path = r"C:\\Users\\Administrator\\Git\\CFturb\\Main_Dimension_Cal\\dataset"
        data_path += r"\\01.csv" if train else r"\\01_test.csv"
        self.csv_data = pd.read_csv(data_path)

    def __getitem__(self, idx):
        data = self.csv_data.iloc[idx, :]
        q = data.iloc[1]
        h = data.iloc[2]
        n = data.iloc[3]
        npsh = data.iloc[4]
        beta2 = data.iloc[9]
        d0k0 = data.iloc[11]
        d2k2d2 = data.iloc[12]
        b2k2b2 = data.iloc[13]
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

