"""
-title:cifar-10 輸入自己的圖片測試
-author:Jessica Cheng(jc)
-memo:需先使用ch8-1訓練完模型後載入使用
-reference: https://github.com/uchidama/CIFAR10-Prediction-In-Keras
"""
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import load_model
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from PIL import ImageOps

im = Image.open('dog_test.jpg') #圖片位置
image = im.resize((32, 32)) #調整圖片大小32x32

resize_frame = np.asarray(image)
#plt.imshow(resize_frame)
#plt.show()

# cifar10 category label name #定義label名稱
cifar10_labels = np.array([
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'])


"""---LOAD MODEL載入模型---"""

# load model
model = load_model('cifar10_model.h5') #for cifar10


"""---PREDICTING預測---"""

def convertCIFER10Data(image):
    img = image.astype('float32')
    img /= 255
    c = np.zeros(32*32*3).reshape((1,32,32,3))
    c[0] = img
    return c


data = convertCIFER10Data(resize_frame)

plt.imshow(resize_frame) #顯示處理後，模型真正看到的圖片
plt.axis('off')

ret = model.predict(data, batch_size=1) 
#print(ret)
print("all probability:")
print("----------------------------------------------")
bestnum = 0.0
bestclass = 0
for n in [0,1,2,3,4,5,6,7,8,9]:
    print("[{}] : {}%".format(cifar10_labels[n], round(ret[0][n]*100,2))) #印出每一項預測機率
    if bestnum < ret[0][n]:
        bestnum = ret[0][n]
        bestclass = n
        
print("----------------------------------------------")

plt.show()
print("highest probability : {}%".format( round(bestnum*100,2) ))
print("the result is a [{}].".format(cifar10_labels[bestclass]))