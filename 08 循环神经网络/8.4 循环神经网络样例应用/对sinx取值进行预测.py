# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/12 -*-


import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.use('Agg')

tf.reset_default_graph()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HIDDEN_SIZE = 30  # LSTM中隐藏节点的个数
NUM_LAYERS = 2  # LSTM的层数

TIMESTEPS = 10  # 循环神经网络训练序列长度
TRAINING_STEPS = 10000  # 训练轮数
BATCH_SIZE = 32  # batch大小

TRAINING_EXAMPLES = 10000  # 训练数据个数
TESTING_EXAMPLES = 1000  # 测试数据个数
SAMPLE_GAP = 0.1  # 采样间隔


# 产生正弦数据
def generate_data(seq):
    X = []
    y = []
    # 用sin函数前面TIMESTEPS个点的信息，预测第i+TIMESTEPS个点的函数值
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# 定义网络结构与优化步骤
def lstm_model(X, y, is_training):
    # 使用多层的LSTM结构
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)  # dynamic_rnn()将多层LSTM结构连接成RNN网络并计算前向传播过程
    output = outputs[:, -1, :]  # 只关注最后一个时刻的输出
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    # 只在训练时计算损失函数和优化步骤。测试时直接返回预测结果
    if not is_training:
        return predictions, None, None
    # 计算损失函数
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    # 创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer='Adagrad', learning_rate=0.1)
    return predictions, loss, train_op  # 得到预测结果、损失函数和训练操作


# 定义训练方法
def train(sess, train_X, train_y):
    # 将训练数据以数据集的方式提供给计算图
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    # X, y只是一个Tensor，并不是实际的值
    X, y = ds.make_one_shot_iterator().get_next()  # 实例化一个Iterator，这个Iterator是一个“one shot iterator”，即只能从头到尾读取一次，get_next()表示从iterator里取出一个元素。
    # 调用模型得到预测结果、损失函数和训练操作
    with tf.variable_scope(name_or_scope='model'):
        predictions, loss, train_op = lstm_model(X, y, is_training=True)
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])  # sess.run([node1, node2, ...])可以传入一个列表，并让它返回多个输出
        if i % 100 == 0:
            print("Train step: " + str(i) + ", loss: " + str(l))


# 定义测试方法
def run_eval(sess, test_X, test_y):
    # 将测试数据以数据集的方式提供给计算图
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    # 调用模型得到计算结果
    with tf.variable_scope(name_or_scope='model', reuse=True):
        predictions, _, _ = lstm_model(X, [0.0], is_training=False)  # 注意这里不需要真是的y值
    # 将预测结果存入一个数组
    predictions = []
    labels = []
    for i in range(TRAINING_EXAMPLES):
        p, l = sess.run([predictions, y])
        predictions.append(p)
        labels.append(l)
    # 计算rmse作为评价指标
    # 通常算法的结果是可以表示向量的数组（即包含两对或以上的方括号形式[[]]），如果直接利用这个数组进行画图可能显示界面为空。
    # 我们可以利用squeeze()函数将表示向量的数组转换为秩为1的数组，这样利用matplotlib库函数画图时，就可以正常的显示结果了。
    predictions = np.array(predictions).squeeze()  # squeeze()只能对维数为1的维度降维
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print('Mean Square Error is: %f' % rmse)
    # 对预测的sin函数进行绘图
    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


# 用正弦函数生成训练和测试数据集合
test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

with tf.Session() as sess:
    # 训练
    train(sess=sess, train_X=train_X, train_y=train_y)
    # 测试
    run_eval(sess=sess, test_X=test_X, test_y=test_y)
