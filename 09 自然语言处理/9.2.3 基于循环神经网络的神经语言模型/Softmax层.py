# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/22 -*-

import tensorflow as tf

# Softmax层是将循环神经网络的输出转化为一个单词表中每个单词的输出概率。
# 为此需要两个步骤：

# 1、使用一个线性映射将循环神经网络的输出映射为一个维度与词汇表大小相同的向量。这一步的输出叫做logits。

# 定义线性映射用到的参数
HIDDEN_SIZE = 12
VOCAB_SIZE = 10000
weight = tf.get_variable(name='weight', shape=[HIDDEN_SIZE, VOCAB_SIZE])
bias = tf.get_variable(name='bias', shape=[VOCAB_SIZE])
# 计算线性映射
# output是RNN的输出，其维度为[batch_size * num_steps, HIDDEN_SIZE]
output = [1, 2]  # 例子
# 将偏差项 bias 加到 value 上面，可以看做是 tf.add 的一个特例，其中 bias 必须是一维的，并且维度和 value 的最后一维相同，数据类型必须和 value 相同。
logits = tf.nn.bias_add(value=tf.matmul(output, weight), bias=bias)

# 2、调用softmax()方法
probs = tf.nn.softmax(logits=logits)  # probs与logits维度相同

# 模型训练通常并不关心概率的具体取值，而更关心最终的log perplexity，这可以作为损失函数
# labels是一个大小为[batch_size * num_steps]的一维数组，它包含每个位置正确的单词编号
# logits的维度是[batch_size * num_steps, HIDDEN_SIZE]，loss的维度与labels相同，代表每个位置上的log perplexity。
targets = [1, 2]
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(tensor=targets, shape=[-1]), logits=logits)  # shape=[-1]即拉平为1维
