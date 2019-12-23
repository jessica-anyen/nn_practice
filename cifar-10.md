# cifar-10

官方code

{% embed url="https://keras.io/examples/cifar10\_cnn/" %}

對官方code做說明的文件

[https://zhuanlan.zhihu.com/p/43024112](https://zhuanlan.zhihu.com/p/43024112)

多用途的code

[https://www.itread01.com/content/1543716189.html](https://www.itread01.com/content/1543716189.html)

各種優化編譯器\(ex: RMSProp、Adam\)

{% embed url="https://blog.csdn.net/willduan1/article/details/78070086" %}

learning\_rate要改用lr:

[https://stackoverflow.com/questions/58028976/typeerror-unexpected-keyword-argument-passed-to-optimizer-learning-rate](https://stackoverflow.com/questions/58028976/typeerror-unexpected-keyword-argument-passed-to-optimizer-learning-rate)

解決問題\(如下\)，需要添加的部分

```text
ValueError: `validation_steps=None` is only valid for a generator based on the `keras.utils.Sequence` class. Please specify `validation_steps` or use the `keras.utils.Sequence` class. 
```

[https://github.com/pierluigiferrari/ssd\_keras/issues/102](https://github.com/pierluigiferrari/ssd_keras/issues/102)

用webcam拍照或錄影，直接使用model，判斷影像是甚麼東西

[https://github.com/uchidama/CIFAR10-Prediction-In-Keras/blob/master/keras\_cifar10\_webcam\_prediction.ipynb](https://github.com/uchidama/CIFAR10-Prediction-In-Keras/blob/master/keras_cifar10_webcam_prediction.ipynb)



