"""
进行模型的训练
"""

from mydataset import get_dataloader
from model_beta2 import PreModel
from torch import optim
import torch
import torch.nn as nn

# 实例化模型，优化器，和损失函数
model = PreModel()
optimizer = optim.Adam(model.parameters(), lr=0.05)


# 进入循环进行训练
def train(epoch):
    train_dataloader = get_dataloader()
    for idx, (q, h, n, beta2, d0k0, d2k2d2, b2k2b2) in enumerate(train_dataloader):
        # 把梯度置零
        optimizer.zero_grad()
        # 计算得预测值
        x = torch.cat((q, h, n), 0)
        x = x.view(1, 3)
        output = model(x)
        # 计算均方差损失
        loss = nn.MSELoss()
        loss1= loss(output, beta2)
        # 反向传播得到损失
        loss1.backward()
        # 参数更新
        optimizer.step()
        # 打印数据
        for index in range(idx):
            print("epoch:{}  idx:{}  loss:{} ".format(epoch, idx, loss1.item()))


if __name__ == '__main__':
    for i in range(10):
        train(i)

