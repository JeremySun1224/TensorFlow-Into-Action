# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/24 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SRC_TRAIN_DATA = './train.en'  # 源语言输入文件
TGR_TRAIN_DATA = './train.zh'  # 目标语言输入文件
CHECKPOINT_PATH = './seq2seq_ckpt'  # checkpoint保存路径

HIDDEN_SIZE = 1024  # LSTM的隐藏层规模
NUM_LAYERS = 2  # 深层循环神经网络中LSTM结构的层数
SRC_VOCAB_SIZE = 10000  # 源语言词汇表大小
TRG_VOCAB_SIZE = 4000  # 目标语言词汇表大小
BATCH_SIZE = 100  # 训练数据batch的大小
NUM_EPOCH = 5  # 使用训练数据的轮数
KEEP_PROB = 0.8  # 节点不被dropout的概率
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True  # 在Softmax层和词向量层之间共享参数

MAX_LEN = 50  # 限定句子的最大单词数量
SOS_ID = 1  # 目标语言词汇表中<sos>的ID


# 使用Dataset从一个文件中读取一个语言的数据。数据的格式为每行的一句话，单词已经转化为单词编号。
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(filenames=file_path)
    # 根据空格将单词编号切分开并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)  # string_split()函数输入数据需为列表，返回的是稀疏矩阵，通过.values获取值
    # 将字符串形式的单词编号转化为整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量并与句子内容一起放入Dataset中
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


# 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行padding和batching操作
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(file_path=src_path)
    trg_data = MakeDataset(file_path=trg_path)
    # 通过zip操作将两个Dataset合并为一个Dataset。现在每个Dataset中每一项数据ds。
    # ds由4个张量组成：ds[0][0]是源句子；ds[0][1]是源句子长度；ds[1][0]是目标句子；ds[1][1]是目标句子长度
    dataset = tf.data.Dataset.zip(datasets=(src_data, trg_data))

    # 刪除内容为空（只包含<eos>的句子和长度过长的句子）
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)

    dataset = dataset.filter(predicate=FilterLength)  # filter()对传入的数据进行条件过滤，predicate为条件过滤函数

    # 解码器需要两种格式的目标句子：1.解码器的输入(trg_input)，形式如同"<sos> X Y Z"；2.解码器的目标输出(trg_label)，形式如同"X Y Z <eos>"。
    # 上面从文件中读到的目标句子是"X Y Z <eos>"的形式，我们需要从中生成'<sos> X Y Z'形式的数据加入到Dataset中
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(map_func=MakeTrgInput)
    # 随机打乱数据
    dataset = dataset.shuffle(buffer_size=10000)
    # 规定填充后的数据维度
    padded_shapes = (
        # [None]源句子是长度未知的向量，[]源句子长度是单个数字
        (tf.TensorShape([None]), tf.TensorShape([])),
        # [None]目标句子（解码器输入）是长度未知的向量，[None]目标句子（解码器目标输出）是长度未知的向量，[]目标句子长度是单个数字。
        (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]))
    )
    # 通过调用padded_batch方法进行batching操作
    batched_dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=padded_shapes)
    return batched_dataset


# 定义NMTModel类来描述模型
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        # 为源语言和目标语言分别定义词向量
        self.src_embedding = tf.get_variable(name='src_emb', shape=[SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(name='trg_emb', shape=[TRG_VOCAB_SIZE, HIDDEN_SIZE])
        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(name='weight', shape=[HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(name='softmax_bias', shape=[TRG_VOCAB_SIZE])

    # 在forward函数中定义模型的前向计算图。src_input, src_size, trg_input, trg_label, trg_size分别是上面MakeSrcTrgDataset函数产生的五种张量。
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]
        # 将输入和输出单词编号转化为词向量
        src_emb = tf.nn.embedding_lookup(params=self.src_embedding, ids=src_input)
        trg_emb = tf.nn.embedding_lookup(params=self.trg_embedding, ids=trg_input)
        # 在词向量上进行dropout
        src_emb = tf.nn.dropout(src_emb, keep_prob=KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, keep_prob=KEEP_PROB)

        # 使用dynamic_rnn构造编码器。
        # 编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态enc_state。
        # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类张量的tuple，每个LSTMStateTuple对应编码器中的一层。
        # enc_outputs是顶层LSTM在每一步的输出，它的维度是[batch_size, max_time, HIDDEN_SIZE]。
        # Seq2Seq模型中不需要用到enc_outputs，而后面介绍的attention模型会用到它。
        with tf.variable_scope(name_or_scope='encoder'):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell, inputs=src_emb, sequence_length=src_size, dtype=tf.float32)

        # 使用dynamic_rnn构造解码器。
        # 解码器取目标句子每个位置的词向量，输出dec_outputs为每一步顶层的LSTM的输出。
        # dec_outputs的维度是[batch_size, max_time, HIDDEN_SIZE]。initial_state=enc_state表示用编码器的输出来初始化第一步的隐藏状态。
        with tf.variable_scope(name_or_scope='decoder'):
            dec_outputs, _ = tf.nn.dynamic_rnn(cell=self.dec_cell, inputs=trg_emb, sequence_length=trg_size, initial_state=enc_state)

        # 计算解码器每一步的log perplexity
        output = tf.reshape(tensor=dec_outputs, shape=[-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(tensor=trg_label, shape=[-1]), logits=logits)

        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练。
        # 通过tf.sequence_mask得到的mask张量，与损失函数结果进行对照相乘，可以去掉无用的损失值，保证了计算的准确性。
        label_weights = tf.sequence_mask(lengths=trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(tensor=label_weights, shape=[-1])
        cost = tf.reduce_sum(input_tensor=loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(input_tensor=label_weights)
        # 定义反向传播操作
        trainable_variables = tf.trainable_variables()
        # 控制梯度大小，定义优化方法与训练步骤
        grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
        grads, _ = tf.clip_by_global_norm(t_list=grads, clip_norm=MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, trainable_variables))
        return cost_per_token, train_op


# 在给定的模型model上训练一个epoch，并返回全局步数。每训练200步便保存一个checkpoint。
def run_epoch(session, cost_op, train_op, saver, step):
    # 训练一个epoch，重复训练步骤直至遍历完Dataset中所有数据。
    while True:
        try:
            # 运行train_op并计算损失值。训练数据在main()函数中以Dataset方式提供
            cost, _ = session.run([cost_op, train_op])
            if step % 100 == 0:
                print('After %d steps, per token cost is %.3f' % (step, cost))
            # 每200步保存一个checkpoint
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
    # 定义训练用的循环神经网络
    with tf.variable_scope(name_or_scope='nmt_model', reuse=None, initializer=initializer):
        train_model = NMTModel()
    # 定义输出数据
    data = MakeSrcTrgDataset(src_path=SRC_TRAIN_DATA, trg_path=TGR_TRAIN_DATA, batch_size=BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()
    # 定义前向计算图，输入数据以张量形式提供给forward函数
    cost_op, train_op = train_model.forward(src_input=src, src_size=src_size, trg_input=trg_input, trg_label=trg_label, trg_size=trg_size)
    # 训练模型
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print('In iteration: %d' % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(session=sess, cost_op=cost_op, train_op=train_op, saver=saver, step=step)


if __name__ == '__main__':
    main()

"""
In iteration: 1
After 0 steps, per token cost is 8.287
After 100 steps, per token cost is 5.893
After 200 steps, per token cost is 5.782
After 300 steps, per token cost is 5.557
After 400 steps, per token cost is 5.505
After 500 steps, per token cost is 5.472
After 600 steps, per token cost is 5.470
After 700 steps, per token cost is 5.048
After 800 steps, per token cost is 5.158
After 900 steps, per token cost is 5.442
After 1000 steps, per token cost is 5.364
After 1100 steps, per token cost is 5.186
After 1200 steps, per token cost is 4.965
After 1300 steps, per token cost is 5.038
After 1400 steps, per token cost is 5.157
After 1500 steps, per token cost is 5.078
After 1600 steps, per token cost is 5.163
After 1700 steps, per token cost is 4.836
After 1800 steps, per token cost is 4.597
After 1900 steps, per token cost is 4.802
After 2000 steps, per token cost is 4.826
需要太长运行时间了，全部训练完可能需要60个小时。
这里只取用第2000步的运行结果进行翻译测试。
"""
