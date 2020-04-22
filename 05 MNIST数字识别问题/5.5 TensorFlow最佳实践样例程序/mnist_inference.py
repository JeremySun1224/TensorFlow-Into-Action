# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/8 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 通过tf.get_variable()函数来获取变量
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(name='weights', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection(name='losses', value=regularizer(weights))
    return weights


# 定义前向传播过程
def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope(name_or_scope='layer1'):
        weights = get_weight_variable(shape=[INPUT_NODE, LAYER1_NODE], regularizer=regularizer)
        biases = tf.get_variable(name='biases', shape=[LAYER1_NODE], initializer=tf.constant_initializer(value=0.0))
        layer1 = tf.nn.relu(features=tf.matmul(input_tensor, weights) + biases)

    # 类似的声明第二层神经网络的变量并完成前向传播过程
    with tf.variable_scope(name_or_scope='layer2'):
        weights = get_weight_variable(shape=[LAYER1_NODE, OUTPUT_NODE], regularizer=regularizer)
        biases = tf.get_variable(name='biases', shape=[OUTPUT_NODE], initializer=tf.constant_initializer(value=0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
