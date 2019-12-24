# 類神經網路 作品紀錄 ann practice

## mnist  by bpn 手寫數字辨識
使用mnist 作為訓練集，環境為python 3.5


### 操作步驟
- 先以 **keras_mnist_test.py**訓練資料 
- 於小畫家手寫一數字，並將圖片存成28x28 
- 以 **load_predict.py** 測試所寫數字，出現結果為: 

`the result is :預測數字`

  
## som 動物圖
參考自[popsom](https://github.com/njali2001/popsom)

-環境限定:
- pandas 0.20.3
- py版本 3.5~3.6

### 操作步驟
- 輸入資料集(矩陣)、設定圖顏色與線顏色(code中有註解)
- 執行popsom.py
- 跑出分群結果



## mnist cnn 手寫數字辨識

### 操作步驟
- 使用**ch8_1.py** 建立模型
- 於小畫家手寫一數字，並將圖片存成28x28 
- 使用 **load_predict.py** 測試所寫數字，出現結果為: 

`the result is :預測數字`

## cifar-10 彩色圖片辨識
10張圖片測試參考自:[使用CNN和CIFAR-10數據集構建圖像分類器的指南](https://kknews.cc/zh-tw/code/voga4ga.html)
輸入自己的圖片改寫自:[CIFAR-10 Prediction In Keras](https://github.com/uchidama/CIFAR10-Prediction-In-Keras)
模型建立參考自:[keras 文件](https://keras.io/examples/cifar10_cnn/)

### 操作步驟
**產生十張圖片測試**
- 使用 **cifar_keras.py** 訓練模型 (耗時約4小時)
- 使用 **cifar_10_test.py** 隨機選取10張圖，檢視測試結果

**使用自己的圖片測試**
- 使用 **cifar_keras.py** 訓練模型 (耗時約4小時)
- 使用 **cifar_10_input_test.py**，輸入圖片路徑，執行可檢視判斷結果

### 執行結果

**十張圖片**
>
![result]( https://github.com/jessica-anyen/nn_practice/blob/master/picture/predit1.png)

**使用自己的圖片**
>
![result2]( https://github.com/jessica-anyen/nn_practice/blob/master/picture/cifar_input_result.PNG)
