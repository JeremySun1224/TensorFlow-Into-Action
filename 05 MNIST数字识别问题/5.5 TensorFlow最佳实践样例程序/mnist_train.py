# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/8 -*-

import os
import tensorflow as tf
import mnist_inference
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = 'model'  # os.path.join()不要加./
MODEL_NAME = 'model.ckpt'


def train(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZATION_RATE)
    y = mnist_inference.inference(input_tensor=x, regularizer=regularizer)
    global_step = tf.Variable(initial_value=0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作及其训练过程
    variable_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(inputs=tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                               decay_steps=mnist.train.num_examples / BATCH_SIZE, decay_rate=LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss, global_step=global_step)
    with tf.control_dependencies(control_inputs=[train_step, variables_averages_op]):  # control_inputs是Operation和Tensor构成的列表
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                print('After %d training step(s), loss on training batch is %g.' % (step, loss_value))
                # 保存当前的模型
                saver.save(sess=sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets(train_dir=r'E:\Data\MNIST', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

"""
Extracting E:\Data\MNIST\train-images-idx3-ubyte.gz
Extracting E:\Data\MNIST\train-labels-idx1-ubyte.gz
Extracting E:\Data\MNIST\t10k-images-idx3-ubyte.gz
Extracting E:\Data\MNIST\t10k-labels-idx1-ubyte.gz
After 1 training step(s), loss on training batch is 2.95635.
After 1001 training step(s), loss on training batch is 0.17592.
After 2001 training step(s), loss on training batch is 0.158896.
After 3001 training step(s), loss on training batch is 0.146929.
After 4001 training step(s), loss on training batch is 0.182898.
After 5001 training step(s), loss on training batch is 0.116287.
After 6001 training step(s), loss on training batch is 0.100854.
After 7001 training step(s), loss on training batch is 0.0845908.
After 8001 training step(s), loss on training batch is 0.0840312.
After 9001 training step(s), loss on training batch is 0.0728436.
After 10001 training step(s), loss on training batch is 0.0663968.
After 11001 training step(s), loss on training batch is 0.0633459.
After 12001 training step(s), loss on training batch is 0.0629149.
After 13001 training step(s), loss on training batch is 0.0574931.
After 14001 training step(s), loss on training batch is 0.0528375.
After 15001 training step(s), loss on training batch is 0.0493423.
After 16001 training step(s), loss on training batch is 0.0441634.
After 17001 training step(s), loss on training batch is 0.0500163.
After 18001 training step(s), loss on training batch is 0.0430624.
After 19001 training step(s), loss on training batch is 0.0435293.
After 20001 training step(s), loss on training batch is 0.0392007.
After 21001 training step(s), loss on training batch is 0.0481328.
After 22001 training step(s), loss on training batch is 0.0400612.
After 23001 training step(s), loss on training batch is 0.0402284.
After 24001 training step(s), loss on training batch is 0.0348671.
After 25001 training step(s), loss on training batch is 0.0346866.
After 26001 training step(s), loss on training batch is 0.036349.
After 27001 training step(s), loss on training batch is 0.0350108.
After 28001 training step(s), loss on training batch is 0.0334691.
After 29001 training step(s), loss on training batch is 0.0340997.
"""
