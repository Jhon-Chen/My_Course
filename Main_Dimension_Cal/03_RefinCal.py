import numpy as np
import pandas as pd
from fractions import Fraction

"""
精算d2
"""

dataset = pd.read_csv("C:\\Users\\Administrator\\Git\\CFturb\\Main_Dimension_Cal\\dataset\\01.csv")


# 精确计算所需参数
class refined_cal(object):

    def __init__(self):
        # 以下都是要预测的参数  q：流量，h：扬程，n：转速，d0：当量直径
        self.q =
        self.h =
        self.n =
        self.d0 =
        # 计算比转速
        a = Fraction(3, 4)
        q1 = np.power(self.q, 0.5)
        h1 = np.power(self.h, a)
        self.ns = np.true_divide(3.65 * self.n * q1, h1)

        # 以下也是需要预测的参数
        self.d2 =
        self.b2 =
        self.efh =
        self.efv =

    # 计算u2
    def u2(self):
        # 计算水头ht
        ht = Fraction(self.h, self.efh)

        # 计算vm2
        phi2 = np.random.uniform(0.8, 0.9)
        a = self.efv * self.d2() * np.pi * self.b2() * phi2
        vm2 = Fraction(self.q, a)

        # 计算sigma
        a = Fraction(np.pi, para_cal.num_blade())
        sigma = 1 - a * np.sin(para_cal.beta2() * np.pi/180)

        # 计算u2
        a1 = Fraction(vm2, 2 * sigma * np.tan(para_cal.beta2() * np.pi/180))
        b1 = np.power(Fraction(vm2, a1), 2)
        c1 = Fraction(9.8 * ht, sigma)
        u2 = a1 + np.power(b1 + c1, 0.5)
        return u2


# 计算出现在的d2与先前出入的d2，并计算他们的误差进行迭代
class err_analysis(object):

    def __init__(self):
        self.d2 = para_cal.d2()
        self.u2 = refined_cal.u2()
        self.n = para_cal.n

        pass

    # 计算出现在的d2_new
    def d2_new(self):
        d2_new = Fraction(60 * self.u2, np.pi * self.n)
        return d2_new
