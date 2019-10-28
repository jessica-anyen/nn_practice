import numpy as np
#import pandas as pd 
from keras.utils import np_utils #匯入keras.utils因為後續要將label標籤轉換為One-hotencoding
from keras.datasets import mnist #匯入Keras的mnist模組
import PIL.Image as Image

#公式定義
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(x,0)


def relu_prime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

class Layer: #定義各層

    def __init__(self, dim, id, act, act_prime, isoutputLayer = False):
        self.weight = 2 * np.random.random(dim) - 1
        #self.weight = np.random.random(dim)
        self.delta = None
        self.A = None
        self.activation = act
        self.activation_prime = act_prime
        self.isoutputLayer = isoutputLayer
        self.id = id


    def forward(self, x):
        z = np.dot(x, self.weight)
        self.A = self.activation(z)
        self.dZ = self.activation_prime(z)

        return self.A

    def backward(self, y, rightLayer):
        if self.isoutputLayer:
            error =  self.A - y
            self.delta = np.atleast_2d(error * self.dZ)
        else:
            self.delta = np.atleast_2d(
                rightLayer.delta.dot(rightLayer.weight.T)
                * self.dZ)
        return self.delta

    def update(self, learning_rate, left_a):
        a = np.atleast_2d(left_a)
        d = np.atleast_2d(self.delta)
        ad = a.T.dot(d)
        self.weight -= learning_rate * ad


class NeuralNetwork:

    def __init__(self, layersDim, activation='tanh'): #定義激活函數
        if activation == 'sigmoid': 
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        elif activation == 'relu':
            self.activation = relu
            self.activation_prime = relu_prime

        self.layers = []
        for i in range(1, len(layersDim) - 1):
            dim = (layersDim[i - 1] + 1, layersDim[i] + 1)
            self.layers.append(Layer(dim, i, self.activation, self.activation_prime))

        dim = (layersDim[i] + 1, layersDim[i + 1])
        self.layers.append(Layer(dim, len(layersDim) - 1, self.activation, self.activation_prime, True))

    def fit(self, X, y, learning_rate=0.1, epochs=10000): #100000
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)


        for k in range(epochs):
            if k % 10000 == 0: print
            'epochs:', k

            i = np.random.randint(X.shape[0])
            a = X[i]

            # compute the feed forward #向前調整
            for l in range(len(self.layers)):
                a = self.layers[l].forward(a)


            # compute the backward propagation #計算倒傳遞階段
            delta = self.layers[-1].backward(y[i], None)

            for l in range(len(self.layers) - 2, -1, -1):
                delta = self.layers[l].backward(delta, self.layers[l+1])


            #update weights #更新權重
            a = X[i]
            for layer in self.layers:
                layer.update(learning_rate, a)
                a = layer.A

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.layers)):
            a = self.layers[l].forward(a)
        return a


            
          


if __name__ == '__main__':

 #   X = np.array([[0, 0],
 #                 [0, 1],
 #                 [1, 0],
 #                 [1, 1]])

 #   y = np.array([0, 1, 1, 0])

# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
   #查看mnist資料集筆數
    #print('train data=', len(X_train))
    #print('test data=', len(X_test))
    #print('x_train_image :', X_train.shape) #x_train_image : (60000, 28, 28)
    #print('y_train_label :', y_train.shape) #共60000張圖片資料，圖片像素28*28
    x_Train = X_train.reshape(60000, 28*28).astype('float32') 
    x_Test = X_test.reshape(10000, 28*28).astype('float32')
    #print('x_train:', x_Train.shape)
    #print('x_test:', x_Test.shape)    
    x_Train_normalization = x_Train / 255
    x_Test_normalization = x_Test / 255
    #print("x_train_normalization:" ,x_Train_normalization[0])
    #print("y_train_label:", y_train[:5])
    y_TrainOneHot = np_utils.to_categorical(y_train)
    y_TestOneHot = np_utils.to_categorical(y_test)

    #  sigmoid
    # 784 input nodes. 3 layers, 300 hide nodes, 10 output nodes
    nn = NeuralNetwork([784, 3, 400, 10], activation='sigmoid')
    X = x_Train_normalization
    y = y_TrainOneHot
    #sequences = [0, 9, 4, 15, 4, 5]
    #print("test ", sequences.index(max(sequences)))
    nn.fit(X, y, learning_rate=0.1, epochs=50000) #5000000 #定義學習率，epoch數量
    X = x_Test_normalization
    
    Yes = 0
    No = 0
    Index1 = 0
    for e in X:
        test7 = nn.predict(e)
        if np.argmax(test7) == y_test[Index1]:
            Yes += 1
        else:
            No += 1
        Index1 += 1
    
    print("Correct rate: ", format(float(Yes)/float(Yes+No),'.2f')) #正確率

    im = Image.open('w_7.bmp') #圖片位置
    nim = im.resize( (28, 28), Image.BILINEAR)
    imgarr = np.array(nim)
    #print(imgarr)
    img1d = imgarr.ravel()
    #print(img1d)
    img1d_normalization = img1d / 255 #不是bmp才需使用
    #print(img1d_normalization)
    test7 = nn.predict(img1d_normalization)
    print("the answer is: ", np.argmax(test7)) #答案