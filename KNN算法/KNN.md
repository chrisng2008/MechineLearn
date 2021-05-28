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



KNN算法是一个不需要训练过程的算法

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

由上述的代码可知，和线性回归等算法不同，KNN是一个不需要训练的算法。

K近邻算法是非常特殊，可以认为是没有模型的算法，但是为了和其他算法统一，可以认为训练数据集就是模型本身。



## scikit-learn 中的kNN

scikit-learn 中封装了相应的方法，我们可以来调库来实现。

```python
from sklearn.neighbors import KNeighborsClassifier

# 创建示例对象
kNN_classifier = KNeighborsClassifier(n_neighbors=6) # 6相当于上述代码中的K
kNN_classifier.fit(X_train,y_train)

# 预测 要传入数组
x_predict = x.reshape(1,-1)
kNN_classifier.predict(x_predict)
```



## 代码封装改进

我们可以将代码封装成类似于scikit-learn中的类一样，方便用户调用

```python
# filename: kNNClassifier
import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier:
    def __init__(self,k):
        '''初始化KNN分类器'''
        assert k >= 1, "k must be valid"
        self.k = k
        self.__x_train = None
        self.__y_train = None
   
	def fit(self,X_train,y_train):
        '''根据训练数据集X_train和y_train训练KNN分类器'''
        self.__X_train = X_train
        self.__y_train = y_train    
        return self
    
    def predict(self,X_predict)
    	'''给定待预测数据集X_predict，返回表示X_predict的结果向量'''
        assert self.__X_train is not None and self.__y_train is not None,\
        "must fit before predict"
        assert X_predict.shape[1] == self.__X_train.shape[1], \
        '''the feature number of X_predict must be equal to X_train'''
    	y_predict = [self.__predict(x) for x in X_predict]
        
        return np.array(y_predict)
        
    def self.__predict(self,x):
        assert x.shape[0] == self.__X_train.shape[1] , \
        "the feature number of x must be equal to X_trian"

        distances = [sqrt(np.sum((X_train-x)**2)) for x_train in x_train]
        nearest = np.argsort(distance)
        topK_y = [y_train[i] for i in nearest[:self.k]]
        Votes = Counter(topK_y)
        return Votes.most_common(1)[0][0]
    
   	def __repr__(self):
        return "KNN(k=%d)" % self.k
    
        
```

对于上述代码，fit其实是多余的，但是为了适应scikit-learn 中的`fit`方法，我们额外写一个与其相对应的方法。

`fit`方法中`return self`虽以后的学习再补充

## 超参数



## 网络搜索与k近邻算法中更多的超参数



## 数据归一化



## scikit-learn 中的Scaler



## 更多有关k近邻算法的思考

