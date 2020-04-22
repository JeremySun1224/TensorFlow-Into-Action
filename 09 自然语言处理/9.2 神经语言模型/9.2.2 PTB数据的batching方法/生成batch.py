# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/21 -*-

import numpy as np
import tensorflow as tf

TRAIN_DATA = '../9.2.1 PTB数据集的预处理/ptb.train'
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35


# 从文件中读取数据并返回包含单词编号的数组
def read_data(file_path):
    with open(file=file_path, mode='r') as fin:
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list


def make_batch(id_list, batch_size, num_step):
    num_batches = (len(id_list) - 1) // (batch_size * num_step)  # 总的batch数量
    # 将数据整理成一个维度为[batch_size, num_batches * num_step]
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    # 沿着第二个维度将数据切分为num_batches个batch，存入一个数组
    data_batches = np.split(ary=data, indices_or_sections=num_batches, axis=1)  # 得到batch

    # 重复上述操作，但是每个位置向右移动一位
    label = np.array(id_list[1: num_batches * batch_size * num_step + 1])
    label = np.reshape(label, newshape=[batch_size, num_batches * num_step])
    label_batches = np.split(ary=label, indices_or_sections=num_batches, axis=1)
    return list(zip(data_batches, label_batches))


"""
make_batch(id_list=read_data(file_path=TRAIN_DATA), batch_size=20, num_step=10)
一个batch的样子：
array([[   7,    2, 1550,  155,   49, 1065, 2254,    5,  637,    3],
       [ 923, 2598,    6, 1613,    7,    1,  437, 1067,    4,    1],
       [   0,   57,  117,    1,  600, 1160,   11,    1,    2,   13],
       [  16,  510, 2525,    0, 1663, 7367,   10,  260,  108,   42],
       [ 184,   11,    1,  277,  109,  413,    5,   12,    3,   21],
       [  31,  284,    7, 4801,  977,  481,    8,    2,   82,    2],
       [  92,   60,  111, 1842,    0,   16,  473,   11, 1221,    1],
       [  23,    2,   44,    0,    1, 2060, 5661,   30,    6, 6331],
       [  39,   13,  857, 1540,   11, 1207, 2450,   44, 2790,    2],
       [   0,   67,  553,   46, 1519,   14, 3701,  127,    0,   14],
       [  10,   46, 1154, 1956, 6817,    1, 2747,  589, 2240, 2945],
       [1646,    4,   39,  257,   23, 3524,   24, 3930,   19,   54],
       [  27, 4076,  535,    5,    3,  672,   20,  346,    0,  100],
       [ 109,    0,    1,  430,   33,  410,    7,    6,  527,    4],
       [ 386,    0,   23, 4577,  328, 1096,  946,   34,    2,    1],
       [  25, 1657,   22,    1,    2,  817,   15,  799, 8435, 1889],
       [   2,    1, 4156,    0,   83,   24,    6, 7079,    2,    4],
       [   4,    1,  185,   26, 9556,   18,  106,  103,   18, 1162],
       [   5, 5323,   81,    4,  130,   80,    9,   35, 1042,    0],
       [ 113,  391,   19,  672, 1944,    2, 1660,    7, 3408,    4]])
"""


def main():
    train_batches = make_batch(id_list=read_data(file_path=TRAIN_DATA), batch_size=TRAIN_BATCH_SIZE, num_step=TRAIN_BATCH_SIZE)
    # 这里插入模型的代码
    pass


if __name__ == '__main__':
    main()
