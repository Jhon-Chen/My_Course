"""
进行模型的测试
"""

from mydataset import get_dataloader
from model_d0k0 import PreModel
import numpy as np
import torch
import torch.nn as nn

# 实例化模型，优化器，和损失函数
model = PreModel()


# 进行测试
def test():
    train_dataloader = get_dataloader()
    total_loss = []
    with torch.no_grad():
        for idx, (q, h, n, beta2, d0k0, d2k2d2, b2k2b2) in enumerate(train_dataloader):
            # 计算得预测值
            x = torch.cat((q, h, n), 0)
            x = x.view(1, 3)
            output = model(x)
            # 计算均方差损失
            loss = nn.MSELoss()
            loss1= loss(output, d0k0)
            total_loss.append(loss1.item())
            # 打印数据
    print("loss:{} ".format(np.mean(total_loss)))


if __name__ == '__main__':
    test()

