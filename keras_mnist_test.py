#模型建立
import os
from tensorflow.examples.tutorials.mnist import *
from keras.models import *
from keras.layers import *
 
# 設置提示訊息:只提示warning 和 Error(1=全部提醒；2=w&e;3=e)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
"""----------載入mnist數據集-------------"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
"one_hot表示用非零即1的数组保存图片表示的数值.比如一个图片上写的是0,内存中不是直接存一个0,而是存一个数组[1,0,0,0,0,0,0,0,0,0].一个图片上面写的是1,那个保存的就是[0,1,0,0,0,0,0,0,0,0]"
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels #各項代號
 
 
"""----------配置模型----------------"""
# 配置結構
model=Sequential() #令模型是多層
 
# 第一隱藏層的配置：輸入784，輸出100
model.add(Dense(100,input_dim=784))
#model.add(Activation("relu"))
model.add(Activation("sigmoid")) #激活函數使用sigmoid較好，keras內建sigmoid公式
model.add(Dropout(0.5)) #dropout正規化
 
# 第二隱藏層的配置：輸入100，輸出100
model.add(Dense(100))
#model.add(Activation("relu"))
model.add(Activation("sigmoid")) #激活函數使用sigmoid較好，keras內建sigmoid公式
model.add(Dropout(0.5)) #dropout正規化
 
# 輸出層的配置：輸入100，輸出10，用了softmax的輸出層結構
model.add(Dense(10))
model.add(Activation("softmax"))
 
# 編譯模型，選定更新方法 #優化器:修剪梯度
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy']) 
 
"""----------訓練模型--------------------"""
print("training starts.....")
model.fit(trX,trY,epochs=20,batch_size=20) #訓練20個epoch，每一個epoch都訓練55000次
 
"""----------評估模型--------------------"""
# 用測試集去評估模型的準確度
accuracy=model.evaluate(teX,teY,batch_size=20)
print('\nTest accuracy:',accuracy[1]) #印出準確率
 
"""----------模型儲存--------------------"""
save_model(model,'my_model') #儲存模型，在程式同一目錄下
print("\n model save")

