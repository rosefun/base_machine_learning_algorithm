from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
## Example 1: iris for classification( 3 classes)
# X, y = datasets.load_iris(return_X_y=True)
# Example 2
X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# my k-NN
class KNN():
    def __init__(self,data,K=3):
        self.X = data[0]
        self.y = data[1]
        self.K = K
    def fit(self, X_test):
        diffX = np.repeat(X_test, len(self.X), axis=0) - np.tile(self.X,(len(X_test),1))
        square_diffX = (diffX**2).sum(axis=1).reshape(len(X_test),len(self.X))
        sorted_index = square_diffX.argsort()
        predict = [0 for _ in range(len(X_test))]
        for i in range(len(X_test)):
            class_count={}
            for j in range(self.K):
                vote_label = self.y[sorted_index[i][j]]
                class_count[vote_label] = class_count.get(vote_label,0) + 1
            sorted_count = sorted(class_count.items(), key=lambda x: x[1],reverse=True)
            predict[i] = sorted_count[0][0]
        return predict
    def predict(self, X_test):
        return self.fit(X_test)
    def score(self,X,y):
        y_pred = self.predict(X)
        return 1 - np.count_nonzero(y-y_pred)/len(y)

knn = KNN((X_train,y_train), K=3)
print(knn.score(X_test,y_test))