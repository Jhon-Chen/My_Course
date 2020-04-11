"""
定义beta2预测模型
"""

import torch.nn as nn
import torch.nn.functional as F


# 定义全连接预测模型
class PreModel(nn.Module):

    def __init__(self):
        super(PreModel, self).__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 1)
        pass

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc1_out_relu = F.relu(fc1_out)
        out = self.fc2(fc1_out_relu)
        return out
