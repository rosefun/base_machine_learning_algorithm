from sklearn import datasets
import numpy as np
## Example 1
# breast cancer for classification(2 classes) X(569, 30) y(569,)
X, y = datasets.load_breast_cancer(return_X_y=True)
y = np.where(y==0, -1, 1)

# my perceptron
class Perceptron():
    def __init__(self):
        self.W = np.ones((len(X[0]),),dtype=float)
        self.b = 0
        self.lr = 0.01
        self.epoch = 100
    def fit(self, X, y):
        for ep in range(self.epoch):
            for i in range(len(X)):
                if y[i]*(np.dot(X[i],self.W)+self.b) <= 0:
                    self.W += self.lr*y[i]*X[i]
                    self.b += self.lr*y[i]
    def predict(self, X):
        return np.where(np.dot(X,self.W)+self.b>0,1,-1)
    def score(self,X,y):
        y_pred = self.predict(X)
        return 1 - np.count_nonzero(y-y_pred)/len(y)
perceptron = Perceptron()
perceptron.fit(X,y)
y_pred = perceptron.predict(X)
print(perceptron.score(X,y))
————————————————
版权声明：本文为CSDN博主「rosefunR」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://rosefun.blog.csdn.net/article/details/104881348