# nn_practice
類神經網路練習

首先使用save_model.py 建立模型，找到模型存在的位置

再使用read_and_ans.py 測試是否可辨識手寫結果

### 測出來的圖片提示

1. 黑底白字辨識度最佳，建議先對圖片處理再轉換
2. 雖然輸入之後仍然會reshape，但是裁減時能盡量接近正方形，判斷效果最佳
3. 判斷pixel值，0.5~0.6適合較粗的筆跡，較細的門檻要調整0.01較適合

### 小筆記
-- reshape vs resize 皆適用於矩陣
reshape: 會返回新值、不影響原有矩陣
resize:會返回新值、並影響原有矩陣

### 參考資料 reference
1.用倒傳遞BPN寫MNIST
https://github.com/TwVenus/BPN-for-MNIST

2. mnist 範例
https://blog.techbridge.cc/2018/01/27/tensorflow-mnist/

3. tensorflow example之後的作業會用到)
https://github.com/aymericdamien/TensorFlow-Examples

4.手寫數字圖片轉為 28 * 28 方法
https://www.zhihu.com/question/55963897
