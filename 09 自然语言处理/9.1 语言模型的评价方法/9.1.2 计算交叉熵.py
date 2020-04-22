# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/19 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 假设词汇表的大小为3，语料包含两个单词‘2 0’
word_labels = tf.constant(value=[2, 0])

predict_logits = tf.constant([
    [2.0, -1.0, 3.0],
    [1.0, 0.0, -0.5]
])

# 使用sparse_softmax_cross_entropy_with_logits()计算交叉熵
loss_sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_labels, logits=predict_logits)
sess = tf.Session()
print(sess.run(loss_sparse))
# 结果为[0.32656264 0.4643688]，这对应两个预测的perplexity损失。

# softmax_cross_entropy_with_logits()需要将预测目标以概率分布的形式给出，而不是word_labels
word_prob_distribution = tf.constant([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0]
])
loss = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution, logits=predict_logits)
print(sess.run(loss))
# 结果为[0.32656264 0.4643688]，这对应两个预测的perplexity损失。

# 由于softmax_cross_entropy_with_logits()允许提供一个概率分布，所有这种方法有更大的自由度，有时可以提高训练效果
word_prob_smooth = tf.constant([
    [0.01, 0.01, 0.98],
    [0.98, 0.01, 0.01]
])
loss_smooth = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution, logits=predict_logits)
print(sess.run(loss_smooth))
# 结果为[0.32656264 0.4643688]，这对应两个预测的perplexity损失。