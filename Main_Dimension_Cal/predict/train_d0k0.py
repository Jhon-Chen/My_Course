"""
进行模型的训练
"""

from mydataset import get_dataloader
from model_d0k0 import PreModel
from torch import optim
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


# 实例化模型，优化器，和损失函数
model = PreModel()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# 保存每一次的损失
loss_list = []


# 进入循环进行训练
def train(epoch):
    train_dataloader = get_dataloader(train=True)
    # bar = tqdm(train_dataloader, total=len(train_dataloader))

    for idx, (q, h, n, npsh, beta2, d0k0, d2k2d2, b2k2b2) in enumerate(train_dataloader):
        # 把梯度置零
        optimizer.zero_grad()
        # 计算得预测值
        q = q.float()
        h = h.float()
        n = n.float()
        npsh = npsh.float()

        x = torch.cat((q, h, n, d0k0), 0)
        # print(x.type())
        # print(x.shape)
        # print(d0k0)
        # x = x.view(9, 3)
        # print(x.shape)

        output = model(x)
        # 计算均方差损失
        loss = nn.MSELoss()
        loss1= loss(output, d0k0)
        # 反向传播得到损失
        loss1.backward()
        loss_list.append(loss1.item())
        # 参数更新
        optimizer.step()
        # 打印数据
        # bar.set_description("epoch:{}  idx:{}  loss:{:.6f}".format(epoch, idx, np.mean(loss_list)))
        if idx % 10 == 0:
            print("epoch:{}  idx:{}  loss:{:.4f}  output:{}".format(epoch, idx, loss1.item(), output))
            # bar.set_description("epoch:{}, idx:{}, loss:{:.6f}".format(epoch, idx, loss.item()))


if __name__ == '__main__':
    for k in range(40):
        train(k)
    plt.figure(figsize=(50, 8))
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
