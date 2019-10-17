""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
ADDJUST BY JESSICA CHENG (jc)
"""

from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import easygui                 #jc add for open img
import numpy as np             #jc add for trans 28*28
#import cv2                     #jc add for trans bw
import pandas as pd            #jc add for cut the white space

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

from PIL import Image          #jc add for open img  #需安裝pillow套件

# Parameters #已建立好的模型位置
model_path = "/tmp/model.ckpt"
 
# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
 
# tf Graph input
X = tf.placeholder("float", [None, num_input])
 
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}
 
# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
 
# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)
 
# Evaluate model
# argmax returns the index with the largest value across axes of a tensor
ans = tf.argmax(prediction, 1)
 
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
 
# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

#選擇要測試的圖片 #jc add for choose img
img_path = easygui.fileopenbox()
print("choose:%s" %img_path)
img=Image.open(img_path).convert('L')  #jc add for 開啟轉灰階

# resize的過程
if img.size[0] != 28 or img.size[1] != 28:
    img = img.resize((28, 28))
    #img = img.reshape((28, 28))
#img.show()

# 暂存像素值的一维數组
arr = []

for i in range(28):
    for j in range(28):
        # mnist 里的颜色是0代表白色（背景），1.0代表黑色
        pixel = 1.0 - float(img.getpixel((j, i)))/255.0
        #jc add for 把有數字的部分凸顯出來
        if (pixel < 0.5): 
            pixel=0 
        else: 
            pixel=1
         
        #pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
        arr.append(pixel)

#arr1 = np.array(arr).reshape((1, 28, 28, 1))
arr1 = np.array(arr).reshape((28, 28))

# Show image that we want to predict
#plt.imshow(mnist.test.images[1].reshape((28, 28))) #選一張要測試的照片
print(arr1.reshape((28, 28)))
plt.imshow(arr1.reshape((28, 28)),cmap = 'gray') #顯示要測試的圖片
plt.show()

# Running a test dataset by loading the model saved earlier
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
 
    saver.restore(sess, model_path)
    print("Model restored from file: %s" % model_path)
    # Calculate the answer for the image
    #print("what: %s" % mnist.test.images[1:2])
    #print("Answer:", sess.run(ans, feed_dict={X: mnist.test.images[1:2]}))  #這邊要改才會答案正確
    print("Answer:", sess.run(ans, feed_dict={X: arr1.reshape((1, 784))}))  #這邊要改(1,784))才會答案正確