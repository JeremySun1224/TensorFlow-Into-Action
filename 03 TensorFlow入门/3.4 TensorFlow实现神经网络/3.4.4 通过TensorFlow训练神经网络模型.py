# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/3/30 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置神经网络参数的过程就是神经网络的训练过程，只有经过有效训练才能解决问题。
# 在神经网络优化算法中，最常用的方法就是反向传播算法。

# 计算一个样例的前向传播结果
# 使用常量进行赋值，一旦数据量增大，这种方法将较为低效
# x = tf.constant([[0.7, 0.9]])
#
# # 通过placeholder实现前向传播算法，placeholder类似占位符
w1 = tf.Variable(tf.random_normal(shape=(2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal(shape=(3, 1), stddev=1, seed=1))
# 定义placeholder作为输入数据的地方，这里维度不一定需要指定
# 这里先给x赋一个float32位的1*2的矩阵作为占位符，具体数据是多少等会在指定
x = tf.placeholder(dtype=tf.float32, shape=(1, 2), name='input')  # (x1, x2)
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 定义会话
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
# print(sess.run(y))  # 需要指定feed_dict，否则会报错
# 这里的{x: [[0.7, 0.9]]}表示给占位符(x1, x2)进行赋值，即x1=0.7，x2=0.9
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))

# 使用TensorFlow表达一个Batch的数据
# 将上面样例程序中的1*2的矩阵输入改为n*2的矩阵就可以得到n个样例的前向传播结果
# 指定n为3
x = tf.placeholder(dtype=tf.float32, shape=(3, 2), name='input')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
init_op = tf.global_variables_initializer()
# 需要运行初始化init_op
sess.run(init_op)
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
sess.close()

# 定义一个简单的损失函数和反向传播算法
# 使用sigmoid函数将y转换为0~1之间
y = tf.sigmoid(y)
# 定义损失函数
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0))
)
# 定义学习率
learning_rate = 0.001
# 定义反向传播算法来优化神经网络的参数
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=cross_entropy)  # 最小化损失函数
