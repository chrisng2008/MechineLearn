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

**更多距离的定义**

- 向量空间余弦相似度 Cosine Similarity
- 调整余弦相似度 Adjusted Cosine Similarity
- 皮尔森相关系数 Pearson Correlation Coefficient
- Jaccard相关系数 Jaccard Coefficient

对于向量空间余弦相似度，我在编写基于协同过滤算法的推荐系统中有涉及，如感兴趣，请自行了解。

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

scikit-learn 中为我们封装了网格搜索，`Grid Search`

对于scikit-learn 中网格搜索的例子，我们用网格搜索

```python
param_grid = {
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range(1,11)]
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(1,11)],
        'p':[i for i in range(1,6)]
    }
}
```

总共的搜索次数为$5*10+10=60$

```python
knn_clf = KNeighborsClassifier()
```

网格搜索

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(knn_clf,param_grid)
grid_search.fit(X_train,y_train)
```

在设置好`param_grid`后我们需要`fit`，这个过程由于需要大量的计算，所以这个过程可能会有一些慢，最后我们调用`best_estimator_`这个属性，就可以搜索出来分类器最佳的超参数。

`GridSearchCV`CV指的是交叉验证。

对于`grid_search`我们需要了解它的其他超参数

`grid_search = GridSearchCV(knn_clf, param_grisd, n_jobs=-1, verbose=2)`

- **n_jobs**是用计算机多核运行的意思，比如2核，那么传入的参数为2，如果想让计算机的所有核都参与运算，那么传入的值为-1
- **verbose**：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。

更多的参数自行了解，`class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch=‘2\*n_jobs’, error_score=’raise’, return_train_score=’warn’)`

## 数据归一化

为什么要进行归一化处理，如果出现部分的数据特别大，那么它将占主导地位

|       | 肿瘤大小(cm) | 发现时间(days) |
| ----- | ------------ | -------------- |
| 样本1 | 1            | 200            |
| 样本2 | 5            | 100            |

在计算距离的时候，我们会发现时间会占主要地位，显然这是不合理的，所以我们要进行数据归一化

|       | 肿瘤大小(cm) | 发现时间(years) |
| ----- | ------------ | --------------- |
| 样本1 | 1            | 0.55            |
| 样本2 | 5            | 0.27            |

以上过程就是归一化。

回到正题，我们谈谈什么是**归一化**：

### 最值归一化

- 解决方案： 将所有的数据映射到同一尺度
- 最值归一化： 把所有的数据映射到0-1之间

$$
x_{scale}=\frac{x-x_{min}}{x_{max}-x_{min}}
$$

适用于分布有明显边界的情况；受outlier影响较大。

### 均值归一化

为了解决上述的情况，我们可以采用**均值方差归一化**(standerdization)

- 均值方差归一化：把所有数据诡异到均值为0方差为1的分布中

适用于数据分布没有明显的边界；有可能存在极端数据值的情况
$$
x_{scale}=\frac{x-x_{mean}}{s}
$$

### 算法实现

```python
import numpy as np

x = np.random.randint(0,100,size=100)
# 最值归一化
(x-np.min(x))/(np.max(x)-np.min(x))

# 均值方差归一化
# 生成一个随机矩阵
x2 = np.random.randint(0,100,(50,2))
# 对第0列进行均值方差归一化
x2[:0] = (x2[:,0]-np.mean(x2[:,0]))/np.std(x2[:,0])
```



## scikit-learn 中的Scaler

### 如何对测试数据集进行归一化

我们可能会用`mean_test`,`std_test`来计算，但是这样是错误的。

正确的做法是用训练数据集的`mean_train`,`std_train`来进行计算

``(x_test-mean_train)/std_train``

**我们为社么要这样做?**

测试数据是模拟真实环境

- 真实环境很有可能无法得到所有测试数据的均值和方差
- 对数据归一化也是算法的一部分

所以我们是需要**保存数据集得到的均值和方差**

### scikit-learn 中使用Scaler

 ```python
 from sklearn.preprocessing import StandarScaler
 
 standarScaler = StandarScaler()
 standarScaler.fit(X_train)
 
 X_mean = standarScaler.mean_	#均值
 X_std = standarScaler.std_	#标准差
 # 使用std_会发出WARNING，因为这种做法以后会弃用，可以使用scale_来替代
 X_scale = standarScaler.scale_	#标准差
 X_train = standarScaler.transform(X_train)
 
 # 对测试集进行归一化
 X_test = standarScaler.transform(X_test)
 ```

### StandarScaler代码封装

```python
import numpy as np

class StandarScaler: 
    def __init__(self):
        self.mean_=None
        self.scale_=None
	def fit(self, X):
        '''根据训练数据集X获得数据的均值和方差'''
        assert X.ndim == 2, "The dimension of X must be 2"
        
        self.mean_ = np.array([np.mean(X[:,i]) for i in range (X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range (X.shape[1])])
        
        return self
    
    def transform(self,X):
		'''将X根据这个StandarScaler进行均值方差归一化处理'''
        assert X.ndim == 2, "The dimension of X must be 2"
       	assert self.mean_ is not None and self.scale_ is not None, \
        	"must fit before transform!"
        assert X.shape[1] == len(self.mean_),\
        "the feature number of X must be equal to mean_ and std_"
       	resX =np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
			resX[:,col] = (X[:,col] - self.mean_[col])/self.scale_[col]
    	return resX
```



## 更多有关k近邻算法的思考

k近邻算法的优点：

- 解决分类问题
- 本身可以解决多分类问题
- 思想简单，效果强大
- 也可以解决回归问题

在scikit-learn中也为我们封装了KNeighborsRegressor这样一个使用k近邻解决回归问题的类

k近邻算法的缺点：

- 效率低下
  - 如果训练集有m个样本，n个特征，则预测每一个新的数据，需要`O(m*n)`的时间复杂度。
  - 优化，使用树结构：KD-Tree, Ball-Tree
- 高度数据相关
  - 如果样本周围有几个错误的预测结果，那么准确率就会受到很大的影响
- 预测结果不具有可解释性
- 维数灾难
  - 随着维度的增加，“看似相近”的两个点之间的距离越来越大
  - 解决方法：降维 (PCA主成分分析等)

 