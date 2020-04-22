# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/2 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义一个变量用于计算滑动平均
v1 = tf.Variable(0, dtype=tf.float32)

# 这里step变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类
ema = tf.train.ExponentialMovingAverage(0.99, step)

# 定义一个更新变量滑动平均的操作
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 获取滑动平均后的值
    print(sess.run([v1, ema.average(v1)]))  # 初始化[0.0, 0.0]
    # 更新变量v1的值到5
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))  # 现在为[5.0, 4.5]

    # 更新step的值为10000
    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    # 再次更新
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

"""
更新的值如下：
[0.0, 0.0]
[5.0, 4.5]
[10.0, 4.555]
[10.0, 4.60945]
"""
