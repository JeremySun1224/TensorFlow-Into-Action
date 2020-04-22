# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/3/31 -*-

import os
import tensorflow as tf
from numpy.random import RandomState

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf.greater和tf.where用法比较
v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

sess = tf.InteractiveSession()
# 比较数据中每一个元素的大小并返回
print(tf.greater(v1, v2).eval())  # [False False  True  True]
# 根据greater的结果，如果是True则选择v1，否则返回v2
print(tf.where(tf.greater(v1, v2), v1, v2).eval())  # [4. 3. 3. 4.]
sess.close()

batch_size = 8

# 两个输入节点
x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x-input')
# 回归问题一般只有一个输出节点
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y-input')

# 前向传播
w1 = tf.Variable(tf.random_normal(shape=(2, 1), stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)

# 生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

# 训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 10000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
        if i % 1000 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('After {i} training step(s), loss on all data is {loss}'.format(i=i, loss=total_loss))
            print(sess.run(w1))
