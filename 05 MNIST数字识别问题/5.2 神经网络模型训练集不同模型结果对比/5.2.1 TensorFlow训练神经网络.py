# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/3 -*-

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


# 给定神经网络的输入和所有参数，计算前向传播过程
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 模型训练过程
def train(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    # 不计算参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)
    # 一般需要将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)  # 动态控制衰减率
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())  # 所有可训练的参数
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))  # 返回每一行中的最大值的位置索引
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 计算所有样例的交叉熵平均值
    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失加上正则化损失
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    decay_steps = mnist.train.num_examples / BATCH_SIZE
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step, decay_steps=decay_steps,
                                               decay_rate=LEARNING_RATE_DECAY, staircase=True)
    # 优化损失函数（即最小化），在进行反向传播算法
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies(control_inputs=(train_step, variables_averages_op)):
        train_op = tf.no_op(name='train')  # 什么都不做，仅做为点位符使用控制边界
    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    # 初始化会话并开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        # 迭代的循环神经网络
        for i in range(TRAINING_STEP):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy using average model is %g' % (i, validate_acc))
            # 生成这一轮使用的一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        # 训练结束后在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), test accuracy using average model is %g' % (TRAINING_STEP, test_acc))


# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类
    mnist = input_data.read_data_sets(r'E:\Data\MNIST', one_hot=True)
    train(mnist=mnist)


# TensorFlow提供的一个主程序入口，tf.app.run()会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()


"""
Extracting E:\Data\MNIST\train-images-idx3-ubyte.gz
Extracting E:\Data\MNIST\train-labels-idx1-ubyte.gz
Extracting E:\Data\MNIST\t10k-images-idx3-ubyte.gz
Extracting E:\Data\MNIST\t10k-labels-idx1-ubyte.gz
After 0 training step(s), validation accuracy using average model is 0.0962
After 1000 training step(s), validation accuracy using average model is 0.9794
After 2000 training step(s), validation accuracy using average model is 0.981
After 3000 training step(s), validation accuracy using average model is 0.9834
After 4000 training step(s), validation accuracy using average model is 0.983
After 5000 training step(s), validation accuracy using average model is 0.9828
After 6000 training step(s), validation accuracy using average model is 0.9826
After 7000 training step(s), validation accuracy using average model is 0.9828
After 8000 training step(s), validation accuracy using average model is 0.984
After 9000 training step(s), validation accuracy using average model is 0.984
After 10000 training step(s), test accuracy using average model is 0.9833
"""