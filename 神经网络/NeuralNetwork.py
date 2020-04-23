import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    np.random.seed(1)
    x_pre = 2*np.random.normal(size = (1000,3))
    y = x_pre[:,0]**2 -0.2*np.cos(3*np.pi*x_pre[:,1]) + x_pre[:,2]
    y = y.reshape(y.shape[0],1)
    return x_pre, y

def standard(data):
    mu = np.mean(data)
    std = np.var(data)
    return (data - mu)/std

class TwoLayerNeuralNetwork():
    def __init__(self,input_dim,hidden_dim,epoch=1000,lr=0.0001):
        self.epoch = epoch
        self.lr = lr
        self.w1,self.w2,self.b1,self.b2 = self.init_args(input_dim,hidden_dim)
        
    def init_args(self,input_dim,hidden_dim):
        np.random.seed(1)
        w1 = 2*np.random.normal(size=(input_dim,hidden_dim))-1
        w2 = 2*np.random.normal(size=(hidden_dim,1))-1
        b1 = np.random.random((1,hidden_dim))
        b2 = np.random.random((1))
        return w1,w2,b1,b2
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
        
    def mse(self,true,pred):
        return np.mean((true-pred)**2)
        
    def fit(self, X, y):
        losses = []
        for j in range(self.epoch):
            l1 = self.sigmoid(np.dot(X, self.w1)+ self.b1)
            l2 = np.dot(l1, self.w2) + self.b2 
            loss = self.mse(y, l2)
            # w2 偏导
            w2_delta = np.dot(l1.T,2*(l2 - y))
            # w1 偏导
            w1_delta = np.dot(X.T, 2*np.dot(l2 - y, self.w2.T)*l1*(1 - l1))
            # b2 偏导
            b2_delta = np.mean(2*(l2 - y))
            # b1 偏导
            b1_delta = np.mean(np.dot(2*(l2 - y), self.w2.T),axis=0)
            self.w2 = self.w2 - self.lr*w2_delta
            self.w1 = self.w1 - self.lr*w1_delta
            self.b2 = self.b2 - self.lr*b2_delta
            self.b1 = self.b1 - self.lr*b1_delta
            losses.append(loss)
        return losses

    def predict(self, X_test):
        l1 = self.sigmoid(np.dot(X_test, self.w1)+ self.b1)
        l2 = np.dot(l1, self.w2) + self.b2 
        return l2
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return self.mse(y_test, y_pred)

if __name__ == '__main__':
    # 数据生成和预处理
    X, Y = generate_data()
    X_std = standard(X)
    X_train = X[:int(len(X)*0.8),:]
    X_test = X[int(len(X)*0.8):,:]
    y_train = Y[:int(len(X)*0.8),:]
    y_test = Y[int(len(X)*0.8):,:]

    model = TwoLayerNeuralNetwork(input_dim = 3,hidden_dim = 10,epoch=400)
    losses = model.fit(X_train, y_train)
    print("train mse:{:.4f}".format(model.score(X_train,y_train)))
    print("test mse:{:.4f}".format(model.score(X_test,y_test)))

    # predict
    y_hap =  model.predict(X_test)
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(len(y_test)),y_test,'r',label='true')
    plt.plot(np.arange(len(y_hap)),y_hap,'g',label='pred')
    plt.legend()
    plt.show()
    # losses
    plt.plot(losses)
    plt.show()