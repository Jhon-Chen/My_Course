import pandas as pd

"""
使用pd的read_excel来解析数据，usecols方法用于解析指定的列。
注意:
不推荐usecols方法使用单个整数值，请在usecols中使用包括从0开始的整数列表，如果usecols是一个整数，那么它将是被认为是暗示解析最后一列。
"""
data = pd.read_excel("E:\\CFturbo机器学习课题\\pump_dataset_npsh.xlsx", usecols=[0, 1, 2, 3, 4, 5, 8, 10, 11, 12])

data.info()

data.to_csv("C:\\Users\\Administrator\\Git\\CFturb\\Main_Dimension_Cal\\dataset\\01.csv")

print(pd.read_csv("C:\\Users\\Administrator\\Git\\CFturb\\Main_Dimension_Cal\\dataset\\01.csv"))


