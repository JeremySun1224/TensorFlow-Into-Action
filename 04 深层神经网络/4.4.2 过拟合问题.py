# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/2 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 获取神经网络边上的权重，并将这个权重的L2正则化损失加入名称为losses的集合中
def get_weight(shape, lambda1):  # 这里的lambda是关键字，最好不用
    # 生成一个向量
    var = tf.Variable(tf.random_normal((shape), dtype=tf.float32))
    tf.add_to_collection(name='losses', value=tf.contrib.layers.l2_regularizer(lambda1)(var))
    return var


x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
batch_size = 8
layer_dimension = [2, 10, 10, 10, 1]  # 定义每一层网络中节点的个数
n_layers = len(layer_dimension)  # 神经网络的层数
cur_layer = x  # 这个变量维护前向传播时最深层的节点，开始的时候就是输入层
in_dimension = layer_dimension[0]  # 当前层的节点个数

# 生成5层全连接的神经网络结构
for i in range(1, n_layers):
    out_dimension = layer_dimension[i]  # 下一层的节点个数
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight(shape=[in_dimension, out_dimension], lambda1=0.001)
    bias = tf.Variable(tf.constant(value=0.1, shape=[out_dimension]))
    # 使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前，将下一层的节点个数更新为当前层的节点个数
    in_dimension = layer_dimension[i]

# 将损失函数加入损失集合
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection(name='losses', value=mse_loss)
loss = tf.add_n(tf.get_collection('losses'))
