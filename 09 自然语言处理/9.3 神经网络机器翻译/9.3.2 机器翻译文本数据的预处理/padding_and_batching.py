# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/23 -*-


import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MAX_LEN = 50  # 限定句子的最大单词数量
SOS_ID = 1  # 目标语言词汇表中<sos>的ID


# 使用Dataset从一个文件中读取一个语言的数据。数据的格式为每行的一句话，单词已经转化为单词编号。
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(filenames=file_path)
    # 根据空格将单词编号切分开并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)  # string_split()函数输入数据需为列表，返回的是稀疏矩阵，通过.values获取值
    # 将字符串形式的单词编号转化为整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.float32))
    # 统计每个句子的单词数量并与句子内容一起放入Dataset中
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


# 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行padding和batching操作
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(file_path=src_path)
    trg_data = MakeDataset(file_path=trg_path)
    dataset = tf.data.Dataset.zip(datasets=(src_data, trg_data))
    # 刪除内容为空（只包含<eos>的句子和长度过长的句子）
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)
    dataset = dataset.filter(predicate=FilterLength)  # filter()对传入的数据进行条件过滤，predicate为条件过滤函数
    # 生成'<sos> X Y Z'形式的数据加入到Dataset中
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(map_func=MakeTrgInput)
    # 随机打乱数据
    dataset = dataset.shuffle(buffer_size=10000)
    # 规定填充后的数据维度
    padded_shapes = (
        (tf.TensorShape([None]), tf.TensorShape([])),
        (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]))
    )
    # 通过调用padded_batch方法进行batching操作
    batched_dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=padded_shapes)
    return batched_dataset
