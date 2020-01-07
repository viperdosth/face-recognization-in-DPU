import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys

image_size = 64
num_channels = 3
images = []

path = './i/WIN_20191207_19_48_37_Pro.jpg'
#path = '../project/verify/10.jpg'
image = cv2.imread(path)
image = cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
images.append(image)
images = np.array(images,dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images,1.0/255.0)
x_batch = images.reshape(1,image_size,image_size,num_channels)

sess = tf.Session()
saver = tf.train.import_meta_graph('./project/model.ckpt.meta')
saver.restore(sess,'./project/model.ckpt')
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")
#output after softmax function

x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")


feed_dict_testing = {x:x_batch}
result = sess.run(y_pred,feed_dict=feed_dict_testing)

res_label = ['qizhang','others']
print(res_label[result.argmax()])
print(result)
