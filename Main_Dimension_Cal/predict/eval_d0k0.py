"""
进行模型的测试
"""

from mydataset import get_dataloader
from model_d0k0 import PreModel
import numpy as np
import torch
import torch.nn as nn
import os
# 实例化模型，优化器，和损失函数
model = PreModel()


# 进行测试
def eval():

    test_dataloader = get_dataloader(train=False)
    total_loss = []
    with torch.no_grad():
        for idx, (q, h, n, npsh, beta2, d0k0, d2k2d2, b2k2b2) in enumerate(test_dataloader):
            # 计算得预测值
            q = q.float()
            h = h.float()
            n = n.float()
            npsh = npsh.float()

            x = torch.cat((q, h, n, npsh), 0)
            if os.path.exists("./model_save/my_model.pkl"):
                my_model = torch.load("./model_save/my_model.pkl")
            output = my_model(x)
            # 计算均方差损失
            loss = nn.MSELoss()
            loss1 = loss(output, d0k0)
            total_loss.append(loss1.item())
            # 打印数据
    print("loss:{} ".format(np.mean(total_loss)))


if __name__ == '__main__':
    eval()

