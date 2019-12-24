"""
-title:cifar-10模型測試
-author:Jessica Cheng(jc)
-memo:需先使用ch8-1訓練完模型後載入使用
-reference: https://kknews.cc/zh-tw/code/voga4ga.html
"""
#測試訓練模型
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import cifar10  #for 16張test

# 設置提示訊息:只提示warning 和 Error(1=全部提醒；2=w&e;3=e) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
"""---------載入已經訓練好的模型---------"""
new_model = load_model('cifar10_model.h5') #for cifar10

"""----- 隨機10張圖看測試準確度-----"""
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test2 = (x_test/255) - 0.5

y_pred_test = new_model.predict_proba(x_test2) 
y_pred_test_classes = np.argmax(y_pred_test, axis=1) 
y_pred_test_max_probas = np.max(y_pred_test, axis=1)

cols = 5 
rows = 2 
fig = plt.figure(figsize=(2 * cols - 2, 3 * rows - 1)) 
for i in range(cols): 
    for j in range(rows): 
        random_index = np.random.randint(0, len(y_test)) 
        ax = fig.add_subplot(rows, cols, i * rows + j + 1) 
        ax.grid('off') 
        ax.axis('off') 
        ax.imshow(x_test[random_index, :]) 
        pred_label = cifar10_classes[y_pred_test_classes[random_index]] 
        pred_proba = y_pred_test_max_probas[random_index] 
        true_label = cifar10_classes[y_test[random_index, 0]] 
       # ax.pred_label(fontsize=5)
        ax.set_title("pred: {}\nscore: {:.3}\ntrue: {}".format(
            pred_label, pred_proba, true_label 
        ), fontsize=8)  #調圖片字形大小
plt.show()



