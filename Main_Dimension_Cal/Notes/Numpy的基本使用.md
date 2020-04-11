# Numpy的基本使用

[toc]



目标：

* 了解Numpy运算速度上的优势
* 知道Numpy的数组内存块风格
* 知道Numpy的并行化运算

## NumPy介绍

* 一个强大的**N维数组**对象
* 支持**大量的数据运算**
* 集成C/C++和Fortran代码的工具
* **众多机器学习框架的基础库**（Scipy/Pandas/scikit-learn/Tensorflow）

### NumPy的特点

为什么NumPy会快？

我们都知道Python作为一个动态语言一大特点就是慢，语言本身的特点我们可以抛开不说，**并且CPython还带有GIL锁，发挥不了多核的优势**。注意Django、Flask或者Scrapy这些框架，其实都是一些基于网络的操作（主要是IO）操作，下图是**开销**按代销排序：

![开销](file:///E:/AI/09-%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98-%E8%83%A1%E6%98%9F%E8%BE%89/day_03_%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E4%B9%8Bpandas/02_%E8%AF%BE%E4%BB%B6/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E7%AC%AC%E4%B8%89%E5%A4%A9%E8%AF%BE%E4%BB%B6/images/%E5%BC%80%E9%94%80.png)

在机器学习中，充满了大量的计算。如果还是使用原来的Python函数或者工具，那么估计在机器学习领域就没有Python什么事了。但是有NumPy就好多了，接下来我们要了解了解NumPy的优势。

### NumPy的数组内存块风格

在numpy当中一个核心就是ndarray，他与Python列表的区别如下：

![numpy内存地址](file:///E:/AI/09-%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98-%E8%83%A1%E6%98%9F%E8%BE%89/day_03_%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E4%B9%8Bpandas/02_%E8%AF%BE%E4%BB%B6/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E7%AC%AC%E4%B8%89%E5%A4%A9%E8%AF%BE%E4%BB%B6/images/numpy%E5%86%85%E5%AD%98%E5%9C%B0%E5%9D%80.png)

从图中我们看出来numpy其实在储存数据的时候，数据与数据的地址是连续的，这样就给我们操作带来了好处，处理速度快。在计算机内存里是存储在一个连续空间上的，而对于这个连续空间，我们如果创建Array的方式不同，在这个连续空间上的排列顺序也不同。

* 创建araay的默认方式是`C-type`以row为主在内存中排列
* 如果是`Fortran`的方式创建的，就是以column为主宰内存中排列

如下图：

![内存排列](file:///E:/AI/09-%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98-%E8%83%A1%E6%98%9F%E8%BE%89/day_03_%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E4%B9%8Bpandas/02_%E8%AF%BE%E4%BB%B6/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E7%AC%AC%E4%B8%89%E5%A4%A9%E8%AF%BE%E4%BB%B6/images/%E5%86%85%E5%AD%98%E6%8E%92%E5%88%97.png)

### NumPy的并行化计算

那么NumPy的第二个特点就是，支持并行化运算，也叫向量化运算。numpy的许多函数不仅是用C实现了，还使用了BLAS（一般Windows下link到MKL的，下link到OpenBLAS）。基本上那些BLAS实现在每种操作上都进行了高度优化，例如使用AVX向量指令集，甚至能比你自己用C实现快上许多，更不要说和用Python实现的比。

也就是说numpy底层使用BLAS做向量，矩阵运算。比如我们刚才提到的房子面积到价格的运算，很容易使用multi-threading或者vectorization来加速。



## 属性

目标：

* 说明数组的属性、形状、类型

### ndarry

NumPy提供了一个**N维数组类型ndarray**，它描述了**相同类型**的item的集合。

![学生成绩数据](file:///E:/AI/09-%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98-%E8%83%A1%E6%98%9F%E8%BE%89/day_03_%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E4%B9%8Bpandas/02_%E8%AF%BE%E4%BB%B6/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E7%AC%AC%E4%B8%89%E5%A4%A9%E8%AF%BE%E4%BB%B6/images/%E5%AD%A6%E7%94%9F%E6%88%90%E7%BB%A9%E6%95%B0%E6%8D%AE.png)

#### 特点

* 每个item都占用相同大小的内存块
* 每个item是由单独的数据类型对象指定的，除了基本类型（整型、浮点数等）之外，数据类型对象还可以表示数据结构。

#### 属性

数组属性反映了数组本身固有的信息：

![image-20200323212622928](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200323212622928.png)

查看各个属性的API：

* 元素类型：`a.dtype`
* 元素个数：`a.size`
* 总字节数：`a.nbytes`
* 一个元素的长度：`a.itemsize`

#### 数组的形状

从数学概念的角度理解即可

#### 数组的类型

![image-20200323213224947](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200323213224947.png)

**创建时指定类型：**

`a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)`

**自定义数据结构：**

通常对于numpy数组来说，存储的都是同一类型的数据。但其实也可以通过np.dtype实现**数据类型对象表示数据结构**。

```python
>>> mytype = np.dtype([('name', np.string_, 10), ('height', np.float64)])
>>> mytype
dtype([('name', 'S10'), ('height', '<f8')])

>>> arr = np.array([('Sarah', (8.0)), ('John', (6.0))], dtype=mytype)
>>> arr
array([(b'Sarah', 8.), (b'John', 6.)],
      dtype=[('name', 'S10'), ('height', '<f8')])
>>> arr[0]['name']
```

> 对于储存复杂关系的数据，我们其实会选择Pandas更加方面的工具，一般用不到。



## 基本操作

目标：

* 理解数组的各种创建方法
* 应用数组的索引机制实现数组的切片获取
* 应用维度变换实现数组的形状改变
* 应用类型变换实现数组类型改变
* 应用数组的转置
* 说明数组的copy作用
* 说明正态分布的平均值和标准差

### 数组

#### 0和1的数组

* `ones(shape[ , dtype, order])`
* `zeros(shape[ , dtype, order])`

#### 从现在有的数据中心创建

* `array(object[ , dtype, copy, order, subok, ndmin])`
* `asarray(a[ , dtype, order])`
* `copy(a[ , order])`

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
# 从现在的数组中创建
a1 = np.array(a)
# 相当于索引的形式，并没有真正的创建一个新的
a2 = np.asarray(a)
```

**关于array和asarray的不同：**

![asarray](file:///E:/AI/09-%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98-%E8%83%A1%E6%98%9F%E8%BE%89/day_03_%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E4%B9%8Bpandas/02_%E8%AF%BE%E4%BB%B6/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%9F%BA%E7%A1%80%E7%AC%AC%E4%B8%89%E5%A4%A9%E8%AF%BE%E4%BB%B6/images/asarray.png)

#### 创建固定范围的数组

* `np.linspace(start, stop, num, endpoint, retstep, dtype)`

  生成等间隔的序列

  * start：序列的起始值
  * **stop：序列的终止值，如果endpoint为true，该值包含于序列中**
  * num：要生成的等间隔样例数量，默认为50
  * endpoint：序列中是否包含stop值，默认为true
  * retstep：如果为true，返回样例，以及连续数字之间的步长
  * dtype：输出ndarray的数据类型

```python
# 生成等间隔的数组
np.linspace(0, 100, 10)
```

* 其他的还有
  * `numpy.array(start, stop. step, dtype)`
  * `numpy.logspace(start, stop, num, endpoint, base, dtype)`

```python
np.arange(10, 50 ,2)
```

#### 创建随机数组

* np.random模块
  * 均匀分布
    * np.random.rand(10)
    * np.random.unitform(0, 100)
    * np.random.randint(100)
  * 正态分布
    * 给定均值 / 标准差 / 维度的正态分布
    * np.random.normal(1.75, 0.2, (3.4))
    * np.random.standard_normal(size=(3, 4))

```python
# 创建均匀分布的数组
# 范围0~1
np.random.rand(10)

# 默认范围一个数
np.random.uniform(0, 100)

#随机整数
np.random.randint(10)

np.random.normal(1.75, 0.1, (10, 10))
```

#### 数组的操作

* 索引

  ```python
  # 二维的数组、两个维度
  stock_day_rise[0, 0:10]
  
  # 三维
  a1 = np.array([[[1, 2, 3], [4, 5, 6]], [[12, 3, 34], [5, 6, 7]]])
  a1[0, 0, 1]
  ```

* 修改形状

  ```python
  # 在转化形状的时候，一定要注意数组的元素匹配
  stock_day_price.reshape([504, 500])
  
  stock_day_rise.resize([504,500])
  
  stock_day_rise.flatten()
  ```

* 修改类型

  ```python
  stock_day_rise.reshape([504, 500]).astype(np.int32)
  ```

* 修改小数位数

  ```python
  np.round(stock_day_rise[:2, :20], 4)
  ```

* 数组转换

  ```python
  # 转置
  stock_day_rise.shape
  stock_day_rise.T.shape
  
  # 转换成bytes
  arr.tostring()
  ```

  



### 逻辑运算

目标：

* 应用数组的同样判断函数
* 应用np.where实现数组的三元运算

#### 逻辑判断

```python
# 逻辑判断
temp > 0.5

# 赋值
temp[temp > 0.5] = 1
```

#### 通用判断函数

```python
# np.all()
# 判断stock_day_rise[0:2, 0:5]是否全部上涨
np.all(stock_day_rise[0:2, 0:5] > 0)

# np.unique()
# 返回新的数组值，不存在重复的值
# 将序列中数值值唯一且不重复的值组成新的序列
change_int = stock_day_rise[0:2, 0:5].astype(int)
np.unique(change_int)
```

#### 三元运算符

通过使用np.where能够进行更加复杂的运算

```python
np.where(temp > 0, 1, 0)

# 复合逻辑需要结合 np.ligical_and 和 np.logical_or 使用
# 且，是则为1，否则为0
np.where(np.logical_and(temp > 0.5, temp < 1), 1, 0)
# 或，是则为1，否则为0
np.where(np.logical_or(temp > 0.5, temp < -0.5), 1, 0)
```

### 统计运算

目标：

* 使用np.max完成最大值计算
* 使用np.min完成最小值计算
* 使用np.mean完成平均值计算
* 使用np.std完成标准差计算
* 使用np.argmax、np.argmin完成最大值最小值的索引

#### 统计指标

在数据挖掘 / 机器学习领域，统计指标的值也是我们分析问题的一种方式。常用指标如下：

![image-20200324174729175](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200324174729175.png)

进行统计的时候，**axis轴的取值并不一定，NumPy中不同的API轴的值都不一样，在这里，axis0代表列，axis1代表行进行统计**



### 数组间运算

目标：

* 说明数组间运算的广播机制
* 知道数组与数之间的运算
* 知道数组与数组之间的运算
* 理解矩阵的特点以及运算规则
* 应用np.matmul实现矩阵运算

#### 数组与数的计算

```python
arr = np.array([[1, 2, 3, 2, 1, 4], [5, 6, 1, 2, 3, 1]])
arr + 1
arr / 2

# 可以对比python列表的运算，看出区别
a = [1, 2, 3, 4, 5]
a * 3
```

#### 数组与数组的运算

```python
arr1 = np.array([[1, 2,], [3, 4]])
arr2 = np.array([[2, 3], [4, 5]])
```

上面并不能进行运算！！！

#### 广播机制

**执行broadcast的前提在于，两个ndarray执行的是element-wise的运算，而不是矩阵乘法的运算，矩阵乘法运算时需要维度之间严格匹配。Broadcast机制是为了方便不同形状array（numpy库的核心数据结构）进行数学运算。**

当操作两个数组时，numpy会逐个比较他们的shape（构成的元组tuple），只有在下述情况下，两个数组才能够进行数组与数组：

* 维度相等
* shape（其中相对于的一个地方为1）

```python
Image (3d array):  256 x 256 x 3
Scale (1d array):              3
Result (3d array): 256 x 256 x 3

A      (4d array):  9 x 1 x 7 x 1
B      (3d array):      8 x 1 x 5
Result (4d array):  9 x 8 x 7 x 5

A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  15 x 3 x 5
B      (1d array):  15 x 1 x 1
Result (2d array):  15 x 3 x 5
```

下面这样的，则不匹配：

```python
A  (1d array): 10
B  (1d array): 12
A  (2d array):      2 x 1
B  (3d array):  8 x 4 x 3
```



### 矩阵运算

矩阵，英文matrix，**和array的区别矩阵必须是2维的，但是array可以是多维的。**

* np.mat( )
  * 将数组转换为矩阵类型

```python
a = np.array([[80,86],
[82,80],
[85,78],
[90,90],
[86,82],
[82,90],
[78,80],
[92,94]])
b = np.array([[0.7],[0.3]])

np.mat(a)
```

#### 矩阵乘法运算

需要行列符合行列的要求！！！

* mp.matmul

```python
a = np.array([[80,86],
[82,80],
[85,78],
[90,90],
[86,82],
[82,90],
[78,80],
[92,94]])
b = np.array([[0.7],[0.3]])

np.matmul(a, b)
```

### 合并分割

目标：

* 应用concatenate、vstack、hstack实现数组合并
* 应用split实现数组的横、纵向分割

#### 用处

实现数据的切分和合并，将数据进行切分合并并处理

#### 合并

* numpy.concatenate((a1, a2, ...), axis=0)
* numpy.hstack(tup)  以列合并
* numpy.vstack(tup)  以行合并

```python
a = stock_day_rise[:2, 0:4]
b = stock_day_rise[10:12, 0:4]

# axis=1时候，按照数组的列方向拼接在一起
# axis=0时候，按照数组的行方向拼接在一起
np.concatenate([a, b], axis=0)

array([[-2.59680892, -2.44345152, -2.15348934, -1.86554389],
       [-1.04230807,  1.33132386,  0.52063143,  0.49936452],
       [-1.3083418 , -1.08059664,  0.60855154,  0.1262362 ],
       [ 0.87602641,  0.07077588, -0.44194904,  0.87074559]])

np.hstack([a,b])
np.vstack([a,b])
```

#### 分割

* numpy.split(ary, indices_or_sections, axis=0)

```python
np.split(ab, 4, axis=0)
```



### IO操作与数据处理

目标：

* 知道numpy文件的读取

#### 问题

大多数数据并不是我们自己构造的，存在文件当中。我们需要工具去获取，但是NumPy其实并不适合去读取数据，这里我们了解相关API，以及Numpy不方便的地方即可。

**暂过。IO操作和数据处理更多的使用Pandas。**



