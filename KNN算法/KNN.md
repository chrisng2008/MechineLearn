[toc]

# KNN (K近邻算法)

两个样本如果足够相似的话，那么它就是属于这个类别的。

 两个样本是否相似，我们是用两样本在特征空间的**距离**来描述的。

## 什么是距离

**欧拉距离：**
$$
\sqrt{\sum_{i=1}^{n}(X_{i}^{(a)}-X_{i}^{(b)})^2}
$$



**曼哈顿距离：**
$$
\sum_{i=1}^{n}|X_i^{(a)}-X_i^{(b)}|
$$
我们将欧拉距离和曼哈顿距离改写一下，可得
$$
曼哈顿距离：(\sum_{i=1}^n|X_i^{(a)}-X_i^{(b)}|)^{\frac{1}{1}}\\
欧拉距离：(\sum_{i=1}^n{|X_i^{(a)}-X_i^{(b)}|}^2)^{\frac{1}{2}}
$$


可以发现他们有统一的形式，那么我们可以推广：
$$
(\sum_{i=1}^n{|X_i^{(a)}-X_i^{(b)}|}^p)^{\frac{1}{p}}
$$
这就是**明可夫斯基距离**(Minkowski Distance)

这就是我们下面**超参数**部分提及的米科夫斯基距离。

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

- k：是与当前结点距离最近的k个结点，选出最好的k个结点后，我们通过出现次数来投票决定该结点的类别。

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

​```python
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
        	"the feature number of X_predict must be equal to X_train"
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

## 判断机器学习算法的性能

### 数据集划分

我们可以先将数据划分成**训练集和测试集**，在训练集中训练，模型训练好后，我们来用测试集来进行算法评估。

常见的方法是调用scikit-learn中的`train_test_split`

我们自己实现`train_test_split`的时候要注意，一定要随机选取，因为训练集中的数据有可能已经经过排序的。我们需要将数据集随机化，但是我们不能直接乱序，因为x和y是一一对应的，对于这种情况，我们可以获取乱序后的下标。下面是代码实现

`train_test_split`

```python
# 得到的是索引后元素的序列。
shuffle_indexes = np.random.permutation(len(X))

test_ratio = 0.2 # 20%的数据用来作为测试集
test_size = int(len(X)*test_ratio)

test_indexes = shuffle_indexes[:test_size] 
train_indexes = shuffle_indexes[test_size:]

X_test = X[test_indexes]
y_test = y[test_indexes]

X_train = X[train_indexes]
y_train = y[train_indexes]
```



### train_test_split 代码封装

```python
import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    '''将数据X和y按照test_ratio分割成X_train, X_test, y_train, y_test'''
    assert X.shape[0] == y.shape[0], \
    	"the size of X must be equal to the size of y"
   	assert 0.0 <= test_ratio <= 1.0, \
    	"test_ratio must be valid"
        
    if seed:
        np.random.seed(seed)
        
   	shuffle_indexes = np.random.permutation(len(X))
    test_size = int(len(X)*test_ratio)
    
    test_indexes = shuffle_indexes[:test_size] 
    train_indexes = shuffle_indexes[test_size:]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    X_train = X[train_indexes]
    y_train = y[train_indexes]
    
    return X_train, X_test, y_train, y_test
```



## 超参数

在scikit-learn中我们传入的参数称为超参数

- **超参数**：在算法运行前需要决定的参数
- **模型参数**：算法过程中学习的参数

* kNN算法没有模型参数
* kNN算法中k是典型的超参数

 ### 寻找好的超参数

- 领域知识
- 经验数值
- 实验搜索 

**寻找最好的k**

```python
%% time
best_p = -1
best_score = 0.0
best_k = -1
for k in range (1,11):
    for p in range(1,6):
        knn_clf = KNeighborsClassifier(n_neighbors=k,weights="distance",p=p)
        knn_clf.fit(X_train,y_train)
        score = knn_clf.score(X_test,y_test)
        if score > best_score:
            best_score = score
            best_k = k
            best_p = p 
print("best_k=",best_k)
print("best_score=",best_score)
print("best_p=",best_p)
```

由于最好的超参数可能出现在当前相对最好的超参数附近，因此我们通常需要在当前最好的超参数的附近继续搜索。

### 超参数： distance

除了超参数k外，还有一个超参数我们往往忽略，那就是距离。如果该结点与结点A的距离小于结点B，那么，结点A的权重应该要大于结点B。我们通常以距离的倒数作为权重。好处是：解决平票的情况。scikit-learn中也有weights这个超参数。

### 超参数：明可夫斯基距离

scikit-learn 中与明可夫斯基距离对应的超参数是p

```python
best_score = 0.0
best_k = -1
for k in range (1,11):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train,y_train)
    score = knn_clf.score(X_test,y_test)
    if score > best_score:
        best_score = score
        best_k = k
print("best_k=",best_k)
print("best_score=",best_score)
```

## 网络搜索与k近邻算法中更多的超参数 



## 数据归一化



## scikit-learn 中的Scaler



## 更多有关k近邻算法的思考

