import pandas as pd
import numpy as np
from fractions import Fraction
from sklearn import preprocessing

"""
定义参数计算类
"""
dataset = pd.read_csv("C:\\Users\\Administrator\\Git\\CFturb\\Main_Dimension_Cal\\dataset\\01.csv")
# dataset['outlet angle[°]'].fillna(dataset['outlet angle[°]'].mean(), inplace=True)


# 数据库所需的几个基础参数的计算
class para_cal(object):

    def __init__(self):
        # q：流量，h：扬程，n：转速
        self.q = dataset['flow[m3/s]']
        self.h = dataset['lift[m]']
        self.n = dataset['rotate speed[r/min]']
        self.dj = dataset['impeller inter diameter[mm]']
        self.dh = dataset['shaft diameter[mm]']
        # 计算当量直径
        self.d0 = np.power(np.power(self.dj, 2) + np.power(self.dh, 2), 0.5)
        # 计算比转速
        a = Fraction(3, 4)
        q1 = np.power(self.q, 0.5)
        h1 = np.power(self.h, a)
        self.ns = np.true_divide(3.65 * self.n * q1, h1)
        self.d2 = dataset['impeller outlet diameter[mm]']
        self.b2 = dataset['outlet width[mm]']
        self.beta2 = dataset['outlet angle[°]']
        pass

    # 计算叶轮进口(当量)直径d0的速度系数k0
    def d0k0(self):
        a = np.true_divide(self.q, self.h)
        b = Fraction(1, 3)
        d0k0 = np.true_divide(self.d0, np.power(a, b))
        return d0k0

    # 计算叶轮出口直径d2的速度系数kd2
    def d2kd2(self):
        a = np.true_divide(self.q, self.h)
        b = Fraction(1, 3)
        d2kd2 = np.true_divide(self.d2, np.power(a, b))
        return d2kd2

    # 计算叶轮出口宽度b2的速度系数kb2
    def b2kb2(self):
        a = np.true_divide(self.q, self.h)
        b = Fraction(1, 3)
        b2kb2 = np.true_divide(self.b2, np.power(a, b))
        return b2kb2

    # 确定叶片数，需要后面再确定！！！
    # 暂定范围
    def num_blade(self):
        num_blade = np.random.randint(5, 7)
        return num_blade

    # 计算水力效率
    def efh(self):
        a = np.true_divide(self.q, self.h)
        gh = 1 + 0.0835 * np.log10(a)
        return gh

    # 计算容积效率
    def efv(self):
        a = Fraction(-2, 3)
        b = 1 + 0.68 * np.power(self.ns, a)
        efv = 1/b
        return efv


# 现在把计算出来的需要预测的参数添加到数据集中
if __name__ == '__main__':
    para_cal = para_cal()
    # print(dataset.info())

    # 添加叶轮进口(当量)直径d0的速度系数k0
    dataset.insert(11, 'd0k0', para_cal.d0k0())

    # 添加叶轮出口直径d2的速度系数kd2
    dataset.insert(12, 'd2k2d2', para_cal.d2kd2())

    # 添加叶轮出口宽度b2的速度系数kb2
    dataset.insert(13, 'b2k2b2', para_cal.b2kb2())

    print(dataset.info())

    # # 对数据进行标准化
    # std = preprocessing.StandardScaler()
    # dataset = std.fit_transform(dataset)
    # dataset = pd.DataFrame(dataset)

    dataset.to_csv("C:\\Users\\Administrator\\Git\\CFturb\\Main_Dimension_Cal\\dataset\\01.csv")
    # dataset = pd.DataFrame(dataset, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    #
    # print(dataset.info())
    # # print(dataset.iloc[:, 2])
