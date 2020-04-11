"""
定义d0k0预测模型
"""

import torch.nn as nn
import torch.nn.functional as F


# 定义全连接预测模型
class PreModel(nn.Module):

    def __init__(self):
        super(PreModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 1)
        pass

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc1_out_relu = F.leaky_relu(fc1_out)
        out = self.fc2(fc1_out_relu)
        return out
