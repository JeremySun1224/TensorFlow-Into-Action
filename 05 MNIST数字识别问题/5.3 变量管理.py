# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/5 -*-


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 通过tf.get_variable()和tf.Variable()创建同一个变量
v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(1.0))
v2 = tf.Variable(tf.constant(value=1.0, shape=[1]), name='v2')
with tf.Session() as sess:
    tf.global_variables_initializer().run()  # 需要初始化
    print(sess.run(v1))
    print(sess.run(v2))
    print(v1)
    print(v2)

"""
如果通过tf.get_variable()获取一个已经创建的变量，注意是已经创建好的变量。
需要通过tf.variable_scope()函数来生成一个上下文管理器，
并明确指定在这个上下文管理器中，tf.get_variable()将直接获取已经生成的变量。
"""
# 在名字为foo的命名空间内创建名字为v的变量
with tf.variable_scope('foo'):
    v = tf.get_variable(name='v', shape=[1], initializer=tf.constant_initializer(1.0))
# 尝试在命名空间foo中再次创建已经存在的v1变量，即会报错
# with tf.variable_scope('foo'):
#     v = tf.get_variable(name='v', shape=[1])
"""
报错：ValueError: Variable foo/v already exists, disallowed. 
Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
"""

# 将参数reuse设置为True，tf.get_variable()函数将直接获取已经声明的变量，而且只能获取已经创建的变量。
with tf.variable_scope('foo', reuse=True):
    v1 = tf.get_variable(name='v', shape=[1])
    print(v1 == v)

# tf.get_variable()函数的嵌套
with tf.variable_scope('root'):
    # 获取当前上下文管理器中reuse参数的取值
    print(tf.get_variable_scope().reuse)  # 输出为False，即最外层的reuse是False
    with tf.variable_scope('foo1', reuse=True):
        print(tf.get_variable_scope().reuse)  # 输出为True
        with tf.variable_scope('foo2'):
            print(tf.get_variable_scope().reuse)  # 未指定reuse值则与外面一层保持一致
    print(tf.get_variable_scope().reuse)  # 退出后reuse值又回到了False

# tf.get_variable()函数管理变量命名空间
v2 = tf.get_variable(name='v', shape=[1])
print(v2.name)  # 输出v:0，v即为变量的名称，:0表示这个变量是生成变量这个变量的第一个结果

# tf.get_variable_scope()中创建的变量，名称前面会加上命名空间的名称
with tf.variable_scope('foo', reuse=True):
    v2 = tf.get_variable(name='v', shape=[1])
    print(v2.name)

# 名称空间嵌套
with tf.variable_scope('foo', reuse=False):
    with tf.variable_scope('bar'):
        v3 = tf.get_variable(name='v', shape=[1])
        print(v3.name)
    v4 = tf.get_variable(name='v1', shape=[1])
    print(v4.name)

# 可以直接通过带命名空间名称的变量名来获取其他命名空间下的变量
with tf.variable_scope('', reuse=True):
    v5 = tf.get_variable(name='foo/bar/v', shape=[1])
    print((v5.name))
    print(v5 == v3)

# 打印结果
"""
[1.]
[1.]
<tf.Variable 'v1:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'v2:0' shape=(1,) dtype=float32_ref>
True
False
True
True
False
v:0
foo/v:0
foo/bar/v:0
foo/v1:0
foo/bar/v:0
True
"""

# 通过tf.variable_scope()和tf.get_variable()函数改进前向传播过程
# MNIST数据集相关常数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络的参数
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEP = 10000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, reuse=False):
    # 定义第一层神经网络的变量和前向传播过程
    # 根据传进来的reuse判断是创建新变量还是使用已经创建好的。
    # 在第一次构造网络时需要创建新的变量，以后每次调用这个函数都使用reuse=True则不需要每次都把变量传进来。
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable(name='weights', shape=[INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 类似地定义第二层神经网络的变量和前向传播结果
    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable(name='weights', shape=[LAYER1_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    # 返回前向传播结果
    return layer2

x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE], name='x-input')
y = inference(input_tensor=x)
