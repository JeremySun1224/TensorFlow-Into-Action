# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/3/29 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 声明一个2*3的矩阵变量的方法
weights = tf.Variable(tf.random_normal(shape=[2, 3], stddev=2))  # 均值为0（默认），标准差为2的满足正态分布的随机数
print(weights)
# 要想查看weights的值，先要初始化
# 法1
# sess = tf.InteractiveSession()
# weights.initializer.run()
# print(sess.run(weights))
# 法2：把想要计算得到的内容放到with里
with tf.Session() as sess:
    weights.initializer.run()
    print(sess.run(weights))

# 偏置项通常使用常数来设置初始值
biases = tf.Variable(tf.zeros([3]))  # 初始值全为0且长度为3的向量
print(biases)
with tf.Session() as sess:
    biases.initializer.run()
    print(sess.run(biases))

# 通过其他变量的初始值来初始化新的变量
w2 = tf.Variable(weights.initialized_value())
w3 = tf.Variable(weights.initialized_value() * 2.0)
print(w2)
print(w3)
with tf.Session() as sess:
    w2.initializer.run()
    w3.initializer.run()
    print(sess.run(w2))
    print(sess.run(w3))


# 如何通过变量实现神经网络的参数并实现前向传播的过程
# 声明w1，w2两个变量，这里还通过seed参数设定了随机种子，这样可以保证每次运行得到的结果都是一样的。
w1 = tf.Variable(tf.random_normal(shape=(2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal(shape=[3, 1], stddev=1, seed=1))  # ()和[]差不多
with tf.Session() as sess:
    w1.initializer.run()
    w2.initializer.run()
    print(sess.run(w1))
    print(sess.run(w2))

x = tf.constant([[0.7, 0.9]])
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 需要完成初始化才可以通过sess.run(y)获取y的值
sess.run(w1.initializer)  # 初始化，给变量赋值
sess.run(w2.initializer)  # 初始化，给变量赋值
print(sess.run(y))
# sess.close()

# 上述方法为逐一初始化过程，比较麻烦。TensorFlow提供了一种更加便捷的变量初始化过程，自动处理变量之间的依赖关系
init_op = tf.global_variables_initializer()
sess.run(init_op)
sess.close()

# 测试assign函数
b = tf.Variable([1, 2, 4])
with tf.Session() as sess:
    b.initializer.run()  # 还是要初始化
    print(sess.run(b))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    sess.run(tf.assign(ref=b, value=[1, 2, 9]))
    print(sess.run(b))