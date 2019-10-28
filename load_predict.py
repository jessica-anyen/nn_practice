#測試訓練模型
import os
import cv2
import numpy as np
from keras.models import load_model
# 設置提示訊息:只提示warning 和 Error(1=全部提醒；2=w&e;3=e) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
"""---------載入已經訓練好的模型---------"""
new_model = load_model('my_model')
 
"""---------用opencv載入一張待測圖片-----"""
# 載入圖片
src = cv2.imread('w_9.png') #使用png或bmp都ok
cv2.imshow("test", src) #顯示圖片
 
# 將圖片轉成28*28的灰階
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
dst = cv2.resize(src, (28, 28), interpolation=cv2.INTER_CUBIC) #將圖片轉為28x28 #interpolation:縮放插值方法
 
# 將灰階圖轉成1*784的能夠輸入的數組
picture = np.zeros((1, 784)) #返回來一個給定形狀和型別的用0填充的陣列
for i in range(0, 28):
    for j in range(0, 28):
        picture[0][28 * i + j] = (255 - dst[i][j])


# 用模型進行預測
y = new_model.predict(picture)
result = np.argmax(y) #argmax:取值範圍的最大值
print("the result is", result) #印出結果
 
cv2.waitKey(2731) #cv2畫面停留時間