# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/3/28 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

# 定义两个向量
a = tf.constant([1.0, 2.0], name='a')  # shape=(2,)，shape表示轴的大小，向量[1.0, 2.0]的轴为1，大小为2，所以shape=(2,)
b = tf.constant([2.0, 3.0, 4.0], name='b')  # shape=(3,)
# 在这里把a和b定义为两个常量(tf.constant)
print(a)
print(b)
# 向量相加
c = tf.constant([2.0, 3.0], name='c')
result = a + c
print(result)
# 要想得到相加的结果，不能简单地直接输出result，而需要先生成一个会话(session)并通过这个会话(session)来计算结果
sess = tf.Session()
print(sess.run(result))
