# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/3 -*-

import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 下载数据集
mnist = input_data.read_data_sets(r'E:\Data\MNIST', one_hot=True)

print('Training data size: {size}'.format(size=mnist.train.num_examples))
print('Validating data size: {size}'.format(size=mnist.validation.num_examples))
print('Testing data size: {size}'.format(size=mnist.test.num_examples))
print('Example training data: {data}'.format(data=mnist.train.images[0]))
print('Example training data label: {label}'.format(label=mnist.train.labels[0]))
print(mnist.train.images[0].shape)

# 从所有数据中读取一小部分作为一个batch
batch_size = 100
xs, ys = mnist.train.next_batch(batch_size=batch_size)
print('X shape: {shape}'.format(shape=xs.shape))
print('Y shape: {shape}'.format(shape=ys.shape))

"""
Extracting E:\Data\MNIST\train-images-idx3-ubyte.gz
Extracting E:\Data\MNIST\train-labels-idx1-ubyte.gz
Extracting E:\Data\MNIST\t10k-images-idx3-ubyte.gz
Extracting E:\Data\MNIST\t10k-labels-idx1-ubyte.gz
Training data size: 55000
Validating data size: 5000
Testing data size: 10000
Example training data label: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
(784,)
X shape: (100, 784)
Y shape: (100, 10)
"""
