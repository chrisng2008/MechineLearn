[toc]

# KNN(K近邻算法)

两个样本如果足够相似的话，那么它就是属于这个类别的。

 两个样本是否相似，我们是用两样本在特征空间的**距离**来描述的。

## KNN基础

这里我们使用的距离为我们常见的**欧拉距离**
$$
\sqrt{(x^a-x^b)^2+(y^a-y^b)^2}\\
\sqrt{(x^a-x^b)^2+(y^a-y^b)^2+(z^a-z^b)^2}\\
\sqrt{(X_{1}^{(a)}-X_{1}^{(b)})^2+(X_2^{(a)}-X_2^{(b)})^2+\dots+(X_n^{(a)}-X_n^{(b)})^2}
$$

为了方便表示
$$
\sqrt{\sum_{i=1}^{n}(X_{i}^{(a)}-X_{i}^{(b)})^2}
$$



## 算法实现

以下过程我们用算法来表示

```python
from math import sqrt
distances = []
for x_train in x_train:
    d = sqrt(np.sum((x_train-x)**2))
    distances.append(d) 
# 也可以直接用生成表达式来表示
distances = [sqrt(np.sum((x_train-x)**2)) for x_train in x_train]
```

  `distance`是x与各个点之间的距离，我们需要做的是排序

```python
# argsort 返回的是下标
nearest = np.argsort(distances)
# 从近到远

K = 6
topK_y = [y_train[i] for i in nearest[:k]]

# 接下来我们需要投票来进行选择
from collections import Counter
Votes = Counter(topK_y)
Votes.most_common(1)[0][0]
predict_y = Votes.most_common(1)[0][0]
```

 

## 代码封装

```python
# filename: KNN_classify

import numpy as np
from math import sqrt
from collections import Counter

def KNN_classify(k, X_train, y_train, x):
    # 设置断言来保证用户的输入正确
    assert 1 <= k <= X_trian.shape[0],"k must be vaild"
    assert X_train.shape[0] == y_trian.shape[0],\
    "the size of X_trian must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
    "the feature number of x must be equal to X_trian"
    
    distances = [sqrt(np.sum((X_train-x)**2)) for x_train in x_train]
    nearest = np.argsort(distance)
    topK_y = [y_train[i] for i in nearest[:k]]
    Votes = Counter(topK_y)
   	
    return Votes.most_common(1)[0][0]

```python
```

