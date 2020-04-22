# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/3/28 -*-

# 计算图是TensorFlow中最基本的概念，TensorFlow中的所有计算都会被转化为计算图上的节点

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1.0, 2.0], name='a')
print(a)
b = tf.constant([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], name='b')  # shape=(3, 2)，因为这个矩阵有3个轴，每个轴的大小为2。
print(b)
c = tf.constant([
    [
        [1.0, 2.0],
        [2.0, 3.0]
    ],
    [
        [4.0, 5.0],
        [6.0, 7.0]
    ]
], name='c')
print(c)  # shape=(2, 2, 2)
d = tf.constant([
    [
        [1.0, 2.0],
        [2.0, 3.0],
        [2.0, 4.0]
    ],
    [
        [4.0, 5.0],
        [6.0, 7.0],
        [2.0, 4.0]
    ],
    [
        [8.0, 9.0],
        [9.0, 0.0],
        [2.0, 4.0]
    ]
], name='d')
print(d)  # shape=(3, 3, 2)
# 所以shape的就是有几个轴然后列出每个轴的大小即可。

# 计算图
sess = tf.Session()
result = sess.run(d)
# print(result)

# print(d.graph)
# print(d.graph is tf.get_default_graph())


# tf.Graph生成新的计算图
g1 = tf.Graph()
# print(g1)
# print(g1.as_default())
with g1.as_default():
    # 在计算图g1中定义变量变量v，并设置初始值为0
    v = tf.get_variable(name='v', shape=[1], initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量v，并设置初始值为1
    v = tf.get_variable(name='v', shape=[1], initializer=tf.ones_initializer)

# 在计算图g1中读取变量v的值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable(name='v')))

# 在计算图g2中读取变量v的值
# 首先要生成一个会话
with tf.Session(graph=g2) as sess:
    # 初始化全局变量
    tf.global_variables_initializer().run()
    # 节省存储空间，共享作用域
    with tf.variable_scope('', reuse=True):
        # 用生成的会话去run，填入run的内容，即想要得到的内容
        print(sess.run(tf.get_variable(name='v')))

# tf.Graph.device函数用来指定运行计算的GPU设备
g = tf.Graph()
# 指定运行的设备
e = tf.constant([1.0, 2.0], name='e')
with g.device('/gpu:0'):
    result = a + e
    print(result)
    with tf.Session() as sess:
        print(sess.run(result))