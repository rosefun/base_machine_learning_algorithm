import sklearn.datasets as datasets
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
class LogisticRegression():
    def __init__(self,alpha=0.01,epochs=3):
        self.W = None 
        self.b = None
        self.alpha = alpha
        self.epochs = epochs
    def fit(self,X,y): 
        # 设定种子
        np.random.seed(10)
        self.W = np.random.normal(size=(X.shape[1]))
        self.b = 0
        for epoch in range(self.epochs):
            if epoch%50 == 0:
                print("epoch",epoch)
            w_derivate = np.zeros_like(self.W)
            b_derivate = 0
            for i in range(len(y)):
                w_derivate += (y[i] - 1/(1+np.exp(-np.dot(X[i],self.W.T)-self.b)))*X[i]
                b_derivate += (y[i] - 1/(1+np.exp(-np.dot(X[i],self.W.T)-self.b)))
            self.W = self.W + self.alpha*np.mean(w_derivate,axis=0)
            self.b = self.b + self.alpha*np.mean(b_derivate)
        return self
    def predict(self,X):
        p_1 = 1/(1 + np.exp(-np.dot(X,self.W) - self.b))
        return np.where(p_1>0.5, 1, 0)
def accuracy(pred, true):
    count = 0
    for i in range(len(pred)):
        if(pred[i] == true[i]):
            count += 1
    return count/len(pred)
def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

if __name__ == "__main__":
    # input datasets 
    digits = datasets.load_breast_cancer()
    X = digits.data
    y = digits.target
    # 归一化
    X_norm = normalize(X)
    X_train = X_norm[:int(len(X_norm)*0.8)]
    X_test = X_norm[int(len(X_norm)*0.8):]
    y_train = y[:int(len(X_norm)*0.8)]
    y_test = y[int(len(X_norm)*0.8):]
    # model 1
    lr = LogisticRegression(epochs=500,alpha=0.03)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    # 评估准确率
    acc = accuracy(y_pred, y_test)
    print("acc", acc)
    # model 2
    clf_lr = LR()
    clf_lr.fit(X_train, y_train)
    y_pred2 = clf_lr.predict(X_test)
    print("acc2", accuracy(y_pred2, y_test))