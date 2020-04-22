# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/3/30 -*-

# 在一个模拟数据集上训练神经网络解决二分类问题
import os
import tensorflow as tf
from numpy.random import RandomState

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal(shape=(2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal(shape=(3, 1), stddev=1, seed=1))

x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x-input')  # 控制最里面的轴的大小要为2，因为有两个x
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y-input')  # 控制最里面的轴的大小要为1，因为有一个输出值

# 前向传播
a = tf.matmul(x, w1)  # 矩阵乘法
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法（优化）
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=cross_entropy)

# 生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# int(False)==0,int(True)==1
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_op)
    # 训练前初始化的值
    # print(sess.run(w1))
    # print(sess.run(w2))
    # 设定训练的轮数
    STEPS = 10000
    for i in range(STEPS):
        # 每次选取batch_size个大小的文件进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        # 通过选取的样本训练神经网络参数并更新参数
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
        # 每隔一段时间计算在所有数据上的交叉熵并输出
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print('After %d training step(s), cross entropy on all data is %g' % (i, total_cross_entropy))  # 当数值特别大的时候, 用幂形式打印


    # 训练后神经网络参数的值
    print(sess.run(w1))
    print(sess.run(w2))
