# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/3/29 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 前向传播算法通过矩阵乘法可以方便表达
x = tf.constant([[1, 3]], name='x', dtype=tf.float32)
print(x)
w1 = tf.constant([[1, 2, 4], [2, 4, 1]], name='w1', dtype=tf.float32)
print(w1)
# 相乘
result = tf.matmul(x, w1)
print(result)
with tf.Session() as sess:
    print(sess.run(result))