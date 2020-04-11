# Pandas基本使用

[toc]

## Pandas介绍

* 以numpy为基础，借力numpy模块在计算方面性能高的优势
* 基于Matplotlib，能够简便的画图
* 独特的数据结构



## Pandas的数据结构

### 三大结构

Pandas有三个数据结构，Series、DataFrame以及Panel。

* Series(一维数据)
* DataFrame(二维数据)
* Panel(三维数据/面板数据)

> 注意：对于Panel，会很少使用，通常会使用MultiIndex这种结构解决三维数据表示问题

使用数据结构的API：

```python
data = pd.DataFrame(data)
```

### 初识DataFrame

DataFrame对象既有行索引，又有列索引：

* index：行索引，表明不同行，横向索引，0轴，axis=0
* columns：列索引，表明不同列，纵向索引，1轴，axis=1

增加行索引：

```python
data = pd.DataFrame(data, index=data_index)
```

增加列索引：

```python
# 这个是用于生成一组连续的时间序列
date_range(start=None, end=None, periods=None, freq='B')
start：开始时间
end：结束时间
period：时间天数
freq：递进单位，默认1天，'B'默认略过周末

# index代表行索引，columns代表列索引
data = pd.DataFrame(data, index=data_index, columns=data_columns)
```

#### DataFrame索引的设置

修改行列索引值：

```python
# 修改行列索引
data.index[499] = '0.9991' # 注意，这样子并无法修改！！！

# 通过整体修改，不能单个赋值
data.index = [i for i in range(500)]
```

重置索引:

```python
# 重置索引
data.reset_index(drop=True)
```

以某列设置为新的索引：

```python
df = pd.DataFrame({'month':[1, 4, 7, 10], 'year':[1, 1, 2, 2], 'sale':[55, 40, 84, 31]})
# 设置新的索引值，但是返回一个新的dataframe
df = df.set_index(['month'])
# 设置多重索引 MultiIndex 的结构
df.set_index(['year', df.index])

# 打印df的索引
df.index
```

> 通过刚才的设置，这样DataFrame就变成了一个具有MultiIndex的DataFrame。后面会详细介绍这样的结构。

### Series结构

series结构只有行结构：

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200321102449900.png" alt="image-20200321102449900" style="zoom:25%;" />

#### 创建series

通过已有数据创建

* 指定内容，默认索引

  ```python
  pd.Series(np.arange(10))
  ```

* 指定索引

  ```python
  pd.Series([a, b, c, d], index=data_index)
  ```

* 通过字典数据创建

  ```python
  pd.Series({'red':100, 'blue':200, 'green':500, 'yellow':1000})
  ```

#### series获取属性和值

* index
* values



## 基本数据操作

* 记忆DataFrame的形状、行列索引名称获取等基本属性
* 应用Series和DataFrame的索引进行切片获取
* 应用sort_index和sort_values实现索引和值的排序

### 索引操作

NumPy当中我们已经讲过使用索引选取序列和切片选择，pandas也支持类似的操作，也可以直接使用列名、行名，甚至组合使用。

pandas的DataFrame的获取有三种形式：

* 直接使用行列索引（先列后行，其他的方法是先行后列）
* 结合loc或者iloc使用索引
* 使用ix组合索引

```python
# 通过行列索引
data['open'][['2018-02-27']]

# 使用loc
# loc：只能指定行列索引的名字
data.loc['2018-02-27':'2018-02-22', 'open']

# 使用iloc可以通过索引的下标去获取
data.iloc[0:100, 0:2].head()

# 使用ix进行下标的名称组合做引
data.ix[0:10, ['open', 'close']]
# 相当于
data[['close', 'open', 'high']][0:3]
```

### 排序

排序有两种形式，一种杜宇索引进行排序，一种对于内容进行排序

* 使用df.sort_values（默认是从小到大）
  * 单个键进行排序
  * 多个键进行排序
* 使用df.sort_index给索引进行排序

```python
# 按照涨跌幅大小进行排序，使用ascending指定按照从大到小的排序
data = data.sort_values(by='[p_change]', ascending=False)

# 按照多个键进行排序
data = data.sort_values(by=['open', 'high'])

# 对索引进行排序
data.sort_index()
```



## 统计分析

目标：

* 使用describe完成综合统计
* 使用max完成最大值的计算
* 使用min完成最小值的计算
* 使用mean完成平均值计算
* 使用std完成标准差计算
* 使用idxmin、idxmax完成最大值最小值的索引
* 使用cumsum等实现累计分析

### 基本统计分析函数

#### 综合分析

```python
# 计算平均值、标准差、最大值、最小值、分位数
data.describe()
```

#### 单个函数分析

* sum：Sum of values
* mean：Mean of values
* mad：Mean absolute deviation
* median：Arithmetic median of values
* min：Minimum
* max：Maximum
* mode：Mode
* abs：Absolute Value
* prod：Product of values
* std：Bessel-corrected sample standard deviation
* var：Unbiased variance
* idxmax：compute the index labels with the maximum
* idxmin：compute the index labels with the minimum

对于单个函数去进行统计的时候，坐标是还是按照这些默认，列`index(axis=0, default)`，行`columns(axis=0)`指定

```python
# 单独计算
data['close'].max()

# 对所有的列进行计算
data.max(0)

# 对所有的行进行计算
data.max(1)

# 求出列最大值的位置
data.idxmax(axis=0)

# 求出列最小值的位置
data.idxmin(axis=0)
```



### 累计统计分析函数

* cumsum：计算前n个数的和
* cummax：计算前n个数的最大值
* cummin：计算前n个数的最小值
* cumprod：计算前n个数的积

以上这些函数可以对series和DataFrame操作：

```python
# 排序之后，进行累计求和
data = data.sort_index()

# 计算累计函数
stock_rise = data['p_change']
stock_rise.cumsum()
```



## 逻辑与算数运算

目标：

* 应用逻辑运算符号实现数据的逻辑筛选
* 应用query实现数据的筛选
* 应用isin实现数据的筛选
* 应用add等实现数据间的加法运算
* 应用apply函数实现数据的自定义处理

### 使用逻辑运算符号进行筛选

如果我们要获取p_change大于2的该股票数据：

```python
# 进行逻辑判断
# 用true false进行标记，逻辑判断的结果可以作为筛选的依据
data[data['p_change'] > 2]
```

### 复合的逻辑

```python
# 完成一个符合逻辑判断，p_change > 2, open > 15
data[(data['p_change'] > 2) & (data['open'] > 15)]
```

使用上面两种方法后，其实会有些麻烦，Pandas还提供了一种非常方便的函数供逻辑查询：

* data.query(query_str)
  * query_str：逻辑判断的字符串

```python
data.query("p_change > 2 & open > 15")
data.query("p_change > turnover")
```

### isin

判断是否存在某个值，从而进行筛选

```python
# 可以指定值进行一个判断，从而进行筛选操作
data[data['turnover'].isin([4.19])]
data[data['turnover'].isin([4.19, 2.39])]
```

### 数学运算

```python
# 进行数学运算，加上具体的一个数字
data['open'].add(1)

# 自己求出每天 close-open 价格差
# 筛选两列数据
close = data['close']
open1 = data['open']
# 默认按照索引对齐，sub表示减
data['m_price_change'] = close.sub(open1)
```

### 自定义运算函数

```python
# 进行apply函数运算，使用lambda函数来定义运算函数
data[['open', 'close']].apply(lambda x: x.max() - x.min(), axis=0)
data[['open', 'close']].apply(lambda x: x.max() - x.min(), axis=1)
```



## 文件读取与存储

目标：

* 了解Pandas的几种文件读取存储操作
* 应用CSV方式和HDF方式实现文件的读取和存储

我们的数据大部分存于文件当中，所以pandas会支持复杂的IO操作，pandas的API支持众多的文件格式，如CSV、SQL、XLS、JSON、HDF5。

> 注：最常永刚的HDF5和CSV文件

### CSV文件

* pandas.read_csv(filepath_or_buffer, sep=',', delimiter=None)
  * filepath_or_buffer：文件路径
  * usecols：指定读取的列名，列表形式
* DataFrame.to_csv(path_or_buf=None, sep=',', columns=None, header=True, index=True, index_label=None, mode='w', encoding=None)
  * path_or_buf：string or file handle, default None
  * sep：character, default ','
  * columns：sequence, optional
  * mode：'w' 重写；'a' 追加
  * index：是否写进 行索引
  * header：boolean or list of string, default True, 是否写进列索引值

```python
# 读取文件
data = pd.read_csv("file_path.csv", usecols=['open', 'close'])

data[:10].to_csv("file_path.csv", columns=['open'], index=False, mode='a', header=False)
```

### HDF文件

* pandas.read_hdf(path_or_buf, key=None, **kwargs)

  从h5文件当中读取数据

  * path_or_buffer：文件路径
  * key：读取的键
  * mode：打开文件的模式
  * return：Theselected object

```python
close = pd.read_hdf("file_path.h5")
a = close[['000001.SZ', '000002.SZ']]

a.to_hdf("file_path", key="x")
b = pd.read_hdf("file_path.h5", key="x")
```

如果读取的时候出现以下的错误：

![readh5](file:///E:/AI/09-%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98-%E8%83%A1%E6%98%9F%E8%BE%89/day_03_%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E4%B9%8Bpandas/02_%E8%AF%BE%E4%BB%B6/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E7%AC%AC%E4%B8%89%E5%A4%A9%E8%AF%BE%E4%BB%B6/images/readh5.png)

需要安装tables模块避免不能读取hdf文件：

```python
pip install tables
```

### 拓展

优先选择使用hdf文件存储：

* hdf在存储的是支持压缩，使用的方式是blosc，这个是速度最快的也是pandas默认支持的
* 使用压缩可以提高磁盘利用率，节省空间
* hdf还是跨平台的，可以轻松迁移到hadoop上



## 高级处理_缺失值处理

目标：

* 说明Pandas的缺失值类型
* 应用replace实现数据的替换
* 应用dropna实现缺失值的删除
* 应用fillna实现缺失值的填充
* 应用isnull判断是否有缺失数据NaN

### 缺失值的处理逻辑

对于NaN的数据，在numpy中我们是如何处理的？在pandas中我们处理起来非常容易。

* 判断数据是否为NaN：pd.isnull(df), pd.notnull(df)

处理方式：

* 存在缺失值NaN，并且是np.nan：
  1. 删除存在缺失值的：dropna(axis='rows')
  2. 替换缺失值：fillna(daf[ ].mean(), inplace=True)
* 不是缺失值nan，有默认标记的

### 存在缺失值

#### 缺失值是 np.nan

* 删除

```python
# pandas删除缺失值，使用dropna的前提是，缺失值的类型必须是np.nan
movie.dropna()
```

* 替换缺失值

```python
# 替换存在缺失值的样本
# 替换成：一般填充平均值或者中位数
movie['Revenue (Millions)'].fillna(movie['Revenue (Millions)'].mean(), inplace=True)

movie['Metascore'].fillna(movie['Metascore'].mean(), inplace=True)
```

#### 缺失值不是nan有自己的标记

<img src="file:///E:/AI/09-%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98-%E8%83%A1%E6%98%9F%E8%BE%89/day_03_%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E4%B9%8Bpandas/02_%E8%AF%BE%E4%BB%B6/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E7%AC%AC%E4%B8%89%E5%A4%A9%E8%AF%BE%E4%BB%B6/images/%E9%97%AE%E5%8F%B7%E7%BC%BA%E5%A4%B1%E5%80%BC.png" alt="问号缺失值" style="zoom:50%;" />

**处理思路分析：**

1. 先替换 ' ? ' 为 np.nan

   df.replace(to_replace=XXX, value=XXX)

2. 再进行缺失值的处理

   ```python
   # 把一些其他值标记的缺失值，替换成np.nan
   wis = wis.replace(to_replace='?', value=np.nan)
   
   wis.dropna()
   ```



## 高级处理_数据离散化

目标：

* 应用cut、qcut实现数据的区间分组
* 应用get_dummies实现数据的哑变量矩阵

### 为什么要进行离散化

连续属性离散化的目的是为了简化数据结构，**数据离散化技术可以用来减少给定连续属性值的个数**。离散化方法经常作为数据挖掘的工具。

### 什么是数据的离散化

**连续属性的离散化就是将连续属性的值域上，将值域划分为若干个离散的区间，最后用不同的符号或整数值代表落在每个子区间中的属性值。**

离散化有很多种方法，这使用一种最简单的方式去操作：

![image-20200321212124202](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200321212124202.png)

使用的工具：

* pd.qcut：对数据进行分组
  * 将数据分组 一般会与value_counts搭配使用，统计每组的个数
* series.value_counts()：统计分组次数

```python
# 自行分组
qcut = pd.qcut(np.abs(p_change), 10)
qcut.value_counts()
```

自定义区间分组：

* pd.cut(data, bins)

```python
# 自己指定分组区间
bins = [-100, -7, -5, -3, 0, 3, 5, 7, 100]
p_counts = pd.cut(p_change, bins)
```



## 高级处理_合并

目标：

* 应用pd.concat实现数据的合并
* 应用pd.join实现数据的合并
* 应用pd.merge实现数据的合并

**如果你的数据由多张表组成，那么有时候需要将不同的内容合并在一起分析**

### pd.concat实现数据合并

* pd.concat([data1, data2], axis=1)
  * 按照行或列进行合并

```python
pd.concat([data, dummaries], axis=1)
```

### pd.merge

* pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
  * 可以指定按照两组数据的共同键值对合并或者左右各自
  * left：a dataframe object
  * right：another dataframe object
  * on：columns(names) to join on. must be found in both the left and right dataframe objects
  * left_on=None, right_on=None：指定左右键

![image-20200321222631612](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200321222631612.png)

**案例：**

* pd.merge合并

  默认内连接

  `result = pd.merge(left, right, on=['key1', 'key2'])`

  ![image-20200321222727806](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200321222727806.png)

* 左连接

  `result = pd.merge(left, right, how='left', on=['key1', 'key2'])`

  ![image-20200321222744326](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200321222744326.png)

* 右连接

  `result = pd.merge(left, right, how='right', on=['key1', 'key2'])`

  ![image-20200321224750729](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200321224750729.png)

* 外连接

  `result = pd.merge(left, right, how='outer', on=['key1', 'key2'])`

  ![image-20200321224835244](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200321224835244.png)

  

## 高级处理_交叉表与透视表

目标：

* 应用crosstab和pivot_table实现交叉表与透视表

###  交叉表与透视表的作用

![image-20200322172135443](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200322172135443.png)

### 使用crosstab交叉表实现上图

* 交叉表：交叉表用于计算一列数据对于另外一列数据的分组个数（寻找两个列之间的关系）

  * pd.crosstab(value1, value2)

  `count = pd.crosstab(data['week'], data['posi_neg'])`

### 使用pivot_table透视表实现上述功能

使用透视表，刚才的过程更加简单

* DataFrame.pivot_table([ ], index=[ ])

  ```python
  # 通过透视表，将整个过程变成更简单一些
  data.pivot_table(['posi_neg'], index=['week'])
  ```



## 高级处理_分组与聚合

目标;

* 应用groupby和聚合函数实现数据的分组与聚合

**分组与聚合通常是分析数据的一种方式，通常与一些统计函数一起使用，查看数据的分组情况**

### 什么是分组与聚合

![分组聚合原理](file:///E:/AI/09-%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98-%E8%83%A1%E6%98%9F%E8%BE%89/day_03_%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E4%B9%8Bpandas/02_%E8%AF%BE%E4%BB%B6/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E7%AC%AC%E4%B8%89%E5%A4%A9%E8%AF%BE%E4%BB%B6/images/%E5%88%86%E7%BB%84%E8%81%9A%E5%90%88%E5%8E%9F%E7%90%86.png)

### 分组API

* DataFrame.groupby(key, as_index=False)
  * key：分组的列数据，可以多个

```python
# 分组，求平均值
col.groupby(['color'])['pricel'].mean()
col['pricel'].groupby(col['color']).mean()

# 分组，数据的结构不变
col.groupby(['color'], as_index=False)['pricel'].mean()
```

