# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/25 -*-


import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 参数设置
SRC_TRAIN_DATA = './train.en'
TRG_TRAIN_DATA = './train.zh'
CHECKPOINT_PATH = './attention_ckpt'

HIDDEN_SIZE = 1024
DECODER_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
BATCH_SIZE = 256
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True

MAX_LEN = 50  # 固定句子的最大单词数量
SOS_ID = 1  # 目标语言词汇表中<sos>的ID


# 读取训练数据并创建Dataset
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    # 根据空格将单词编号切分开并放入一个一维向量。
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数。
    dataset = dataset.map(
        lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中。
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    # 首先分别读取源语言数据和目标语言数据。
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)

    dataset = dataset.filter(FilterLength)

    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(MakeTrgInput)
    dataset = dataset.shuffle(10000)
    # 规定填充后输出的数据维度。
    padded_shapes = (
        (tf.TensorShape([None]),  # 源句子是长度未知的向量
         tf.TensorShape([])),  # 源句子长度是单个数字
        (tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度未知的向量
         tf.TensorShape([])))  # 目标句子长度是单个数字
    # 调用padded_batch方法进行batching操作。
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


# 定义翻译模型
class NMTModel(object):
    def __init__(self):
        # 定义编码器和解码器所用的LSTM结构
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE)
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE)
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE) for _ in range(DECODER_LAYERS)])
        # 定义词向量
        self.src_embedding = tf.get_variable(name='src_emb', shape=[SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(name="trg_emb", shape=[TRG_VOCAB_SIZE, HIDDEN_SIZE])
        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(name="weight", shape=[HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(name="softmax_bias", shape=[TRG_VOCAB_SIZE])

    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]
        # 将输入和输出单词编号转为词向量
        src_emb = tf.nn.embedding_lookup(params=self.src_embedding, ids=src_input)
        trg_emb = tf.nn.embedding_lookup(params=self.trg_embedding, ids=trg_input)
        # 在词向量上进行dropout
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        with tf.variable_scope(name_or_scope='encoder'):
            # 构造编码器时，使用bidirectional_dynamic_rnn构造双向循环网络。
            # 双向循环网络的顶层输出enc_outputs是一个包含两个张量的tuple，每个张量的维度都是[batch_size, max_time, HIDDEN_SIZE]，代表两个LSTM在每一步的输出。
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.enc_cell_fw, cell_bw=self.enc_cell_bw, inputs=src_emb, sequence_length=src_size, dtype=tf.float32
            )
            # 将两个LSTM的输出拼接成一个张量
            enc_outputs = tf.concat(values=[enc_outputs[0], enc_outputs[1]], axis=-1)

        with tf.variable_scope(name_or_scope='decoder'):
            # 选择注意力权重的计算模型。BahdanauAttention是使用一个隐藏层的前馈神经网络。
            # memory_sequence_length是一个维度为[batch_size]的张量，代表batch中每个句子的长度，Attention需要根据这个信息把填充位置的注意力权重设置为0。
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_SIZE, enc_outputs, memory_sequence_length=src_size)
            # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高层的循环神经网络
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell, attention_mechanism, attention_layer_size=HIDDEN_SIZE)
            # 使用attention_cell和dynamic_rnn构造解码器。这里没有指定init_state，也就是没有使用编码器的输出来初始化输入，而完全依赖注意力作为信息来源
            dec_outputs, _ = tf.nn.dynamic_rnn(cell=attention_cell, inputs=trg_emb, sequence_length=trg_size, dtype=tf.float32)

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


# 训练过程
def run_epoch(session, cost_op, train_op, saver, step):
    # 训练一个epoch，重复训练步骤直至遍历完Dataset中所有数据。
    while True:
        try:
            # 运行train_op并计算损失值。训练数据在main()函数中以Dataset方式提供
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print('After %d steps, per token cost is %.3f' % (step, cost))
            # 每200步保存一个checkpoint
            if step % 100 == 0:
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
    data = MakeSrcTrgDataset(src_path=SRC_TRAIN_DATA, trg_path=TRG_TRAIN_DATA, batch_size=BATCH_SIZE)
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
