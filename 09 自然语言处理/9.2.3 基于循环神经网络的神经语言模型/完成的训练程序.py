# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/22 -*-


# 双层的LSTM作为循环神经网路主体并共享Embedding层和Softmax层的参数
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = '../9.2.1 PTB数据集的预处理'
TRAIN_DATA = os.path.join(path, 'ptb.train')
EVAL_DATA = os.path.join(path, 'ptb.valid')
TEST_DATA = os.path.join(path, 'ptb.test')
HIDDEN_SIZE = 300
NUM_LAYERS = 2
VOCAB_SIZE = 10000  # 词典规模
TRAIN_BATCH_SIZE = 20  # 训练数据batch的大小
TRAIN_NUM_STEP = 35  # 训练数据截断长度

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 5
LSTM_KEEP_PROB = 0.9  # LSTM节点不被dropout的概率
EMBEDDING_KEEP_PROB = 0.9  # 词向量不被dropout的概率
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
SHARE_EMB_AND_SOFTMAX = True  # 在softmax层和词向量层之间共享参数


# 通过一个PTBModel类来描述模型，这样方便维护循环神经网络中的状态
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps
        # 定义每一步的输入和预期输出
        self.input_data = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_steps])
        self.targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_steps])
        # 定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cell = [
            tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE),
                                          output_keep_prob=dropout_keep_prob)
            for _ in range(NUM_LAYERS)
        ]
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cell)
        # 初始化最初的状态，即全零的向量。这个向量只在每个epoch初始化第一个batch的时候使用
        self.initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        # 定义单词的词向量矩阵
        embedding = tf.get_variable(name='embedding', shape=[VOCAB_SIZE, HIDDEN_SIZE])
        # 将输入向量转化为词向量
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        # 只在训练时dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, keep_prob=EMBEDDING_KEEP_PROB)
        # 定义输出列表。在这里先将不同时刻LSTM结构的输出收集起来再一起提供给softmax层
        outputs = []
        state = self.initial_state
        with tf.variable_scope(name_or_scope='RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs=inputs[:, time_step, :], state=state)
                outputs.append(cell_output)
        # 把输出队列展开成[batch, hidden_size*num_steps]的形状，然后在reshape成[batch*num_steps, hidden_size]的形状
        output = tf.reshape(tensor=tf.concat(values=outputs, axis=1), shape=[-1, HIDDEN_SIZE])  # 把tensor拉平
        # Softmax层：将RNN在每个位置上的输出转化为各个单词的logits
        # not None == not False == not '' == not 0 == not [] == not {} == not ()
        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable(name='weight', shape=[HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable(name='bias', shape=[VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias  # [-1, HIDDEN_SIZE]*[HIDDEN_SIZE, VOCAB_SIZE]=[-1, VOCAB_SIZE]
        # 定义交叉熵损失函数和平均损失
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]), logits=logits)
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
        # 只在训练模型时定义反向传播操作
        if not is_training: return
        trainable_variables = tf.trainable_variables()  # 取出可以训练的所有参数集合
        # 控制梯度大小，定义优化方法和训练步骤。clip_by_global_norm()可以被用来解决梯度消失或梯度爆炸，解析：https://www.cnblogs.com/marsggbo/p/10055760.html
        # Gradient Clipping的引入是为了处理gradient explosion或者gradients vanishing的问题。当在一次迭代中权重的更新过于迅猛的话，
        # 很容易导致loss divergence。Gradient Clipping的直观作用就是让权重的更新限制在一个合适的范围。
        # clip_norm是截取的比率。如果梯度平方和global_norm超过我们指定的clip_norm，那么就对梯度进行缩放，否则就按照原本的计算结果。
        grads, _ = tf.clip_by_global_norm(t_list=tf.gradients(ys=self.cost, xs=trainable_variables), clip_norm=MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        # 由于apply_gradients()函数接收的是一个(梯度张量, 变量)tuple列表，所以要将梯度列表和变量列表进行捉对组合,用zip函数。
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity
def run_epoch(session, model, batches, train_op, output_log, step):
    # 计算平均perplexity的辅助变量
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    # 训练一个epoch
    for x, y in batches:
        # 在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的就是下一个单词为给定单词的概率
        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op],
            {
                model.input_data: x,
                model.targets: y,
                model.initial_state: state
            }
        )
        total_costs += cost
        iters += model.num_steps
        # 只在训练时输出日志
        if output_log and step % 100 == 0:
            print('After %d steps, perplexity is %.3f' % (step, np.exp(total_costs / iters)))
        step += 1
    # 返回给定模型在给定数据上的perplexity值
    return step, np.exp(total_costs / iters)


# 从文件中读取数据并返回包含单词编号的数组
def read_data(file_path):
    with open(file=file_path, mode='r') as fin:
        # 把整个文档读成一个长字符串
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]  # 将读取的单词编号转化为整数
    return id_list


def make_batch(id_list, batch_size, num_step):
    # 计算总的batch数量，每个batch中包含的单词数量是batch_size * num_step
    num_batches = (len(id_list) - 1) // (batch_size * num_step)
    # 将数据整理成一个维度为[batch_size, num_batches * num_step]的二维数组
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    # 沿着第二个维度将数据切分为num_batches个batch，存入一个数组
    data_batches = np.split(ary=data, indices_or_sections=num_batches, axis=1)  # 得到batch

    # 重复上述操作，但是每个位置向右移动一位。这里得到的是RNN每一步输出所需要预测的下一个单词。
    label = np.array(id_list[1: num_batches * batch_size * num_step + 1])
    label = np.reshape(label, newshape=[batch_size, num_batches * num_step])
    # np.split()把一个数组从左到右按顺序切分
    label_batches = np.split(ary=label, indices_or_sections=num_batches, axis=1)
    # 返回一个长度为num_batches的数组，其中每一项包括一个data矩阵和一个label矩阵。
    return list(zip(data_batches, label_batches))


def main():
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
    # 定义训练所用的循环神经网络模型
    with tf.variable_scope(name_or_scope='language_model', reuse=None, initializer=initializer):
        train_model = PTBModel(is_training=True, batch_size=TRAIN_BATCH_SIZE, num_steps=TRAIN_NUM_STEP)
    # 定义测试所用的循环神经网络模型。它与train_model共用参数，但是没有dropout
    with tf.variable_scope(name_or_scope='language_model', reuse=True, initializer=initializer):
        eval_model = PTBModel(is_training=False, batch_size=EVAL_BATCH_SIZE, num_steps=EVAL_NUM_STEP)
    # 训练模型
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        train_batches = make_batch(id_list=read_data(file_path=TRAIN_DATA), batch_size=TRAIN_BATCH_SIZE, num_step=TRAIN_NUM_STEP)
        eval_batches = make_batch(id_list=read_data(file_path=EVAL_DATA), batch_size=EVAL_BATCH_SIZE, num_step=EVAL_NUM_STEP)
        test_batches = make_batch(id_list=read_data(file_path=TEST_DATA), batch_size=EVAL_BATCH_SIZE, num_step=EVAL_NUM_STEP)

        step = 0
        for i in range(NUM_EPOCH):  # 遍历每一个EPOCH
            print('In iteration: %d' % (i + 1))
            step, train_pplx = run_epoch(session=session, model=train_model, batches=train_batches, train_op=train_model.train_op, output_log=True, step=step)
            print('Epoch: %d Train Perplexity: %.3f' % (i + 1, train_pplx))
            _, eval_pplx = run_epoch(session=session, model=eval_model, batches=eval_batches, train_op=tf.no_op(), output_log=False, step=0)
            print('Epoch: %d Eval Perplexity: %.3f' % (i + 1, eval_pplx))
        _, test_pplx = run_epoch(session=session, model=eval_model, batches=test_batches, train_op=tf.no_op(), output_log=False, step=0)
        print('Test Perplexity: %.3f' % test_pplx)


if __name__ == '__main__':
    main()


"""
In iteration: 1
After 0 steps, perplexity is 10011.576
After 100 steps, perplexity is 1701.493
After 200 steps, perplexity is 1160.050
After 300 steps, perplexity is 915.573
After 400 steps, perplexity is 753.849
After 500 steps, perplexity is 643.022
After 600 steps, perplexity is 568.309
After 700 steps, perplexity is 511.117
After 800 steps, perplexity is 460.702
After 900 steps, perplexity is 423.600
After 1000 steps, perplexity is 396.849
After 1100 steps, perplexity is 369.976
After 1200 steps, perplexity is 348.954
After 1300 steps, perplexity is 329.079
Epoch: 1 Train Perplexity: 325.959
Epoch: 1 Eval Perplexity: 182.176
In iteration: 2
After 1400 steps, perplexity is 177.186
After 1500 steps, perplexity is 163.316
After 1600 steps, perplexity is 165.881
After 1700 steps, perplexity is 163.095
After 1800 steps, perplexity is 158.512
After 1900 steps, perplexity is 156.292
After 2000 steps, perplexity is 154.633
After 2100 steps, perplexity is 149.849
After 2200 steps, perplexity is 146.883
After 2300 steps, perplexity is 145.705
After 2400 steps, perplexity is 143.446
After 2500 steps, perplexity is 140.597
After 2600 steps, perplexity is 137.254
Epoch: 2 Train Perplexity: 136.696
Epoch: 2 Eval Perplexity: 132.448
In iteration: 3
After 2700 steps, perplexity is 119.820
After 2800 steps, perplexity is 105.286
After 2900 steps, perplexity is 112.213
After 3000 steps, perplexity is 110.283
After 3100 steps, perplexity is 109.550
After 3200 steps, perplexity is 109.482
After 3300 steps, perplexity is 108.857
After 3400 steps, perplexity is 106.933
After 3500 steps, perplexity is 105.039
After 3600 steps, perplexity is 104.630
After 3700 steps, perplexity is 104.546
After 3800 steps, perplexity is 102.577
After 3900 steps, perplexity is 100.733
Epoch: 3 Train Perplexity: 100.400
Epoch: 3 Eval Perplexity: 115.968
In iteration: 4
After 4000 steps, perplexity is 99.202
After 4100 steps, perplexity is 84.197
After 4200 steps, perplexity is 89.764
After 4300 steps, perplexity is 89.601
After 4400 steps, perplexity is 88.818
After 4500 steps, perplexity is 88.353
After 4600 steps, perplexity is 87.971
After 4700 steps, perplexity is 87.161
After 4800 steps, perplexity is 85.830
After 4900 steps, perplexity is 85.394
After 5000 steps, perplexity is 85.721
After 5100 steps, perplexity is 84.364
After 5200 steps, perplexity is 83.455
After 5300 steps, perplexity is 82.986
Epoch: 4 Train Perplexity: 82.960
Epoch: 4 Eval Perplexity: 109.462
In iteration: 5
After 5400 steps, perplexity is 74.190
After 5500 steps, perplexity is 75.394
After 5600 steps, perplexity is 78.645
After 5700 steps, perplexity is 76.570
After 5800 steps, perplexity is 75.373
After 5900 steps, perplexity is 75.519
After 6000 steps, perplexity is 75.624
After 6100 steps, perplexity is 74.372
After 6200 steps, perplexity is 74.067
After 6300 steps, perplexity is 74.601
After 6400 steps, perplexity is 73.963
After 6500 steps, perplexity is 73.259
After 6600 steps, perplexity is 72.354
Epoch: 5 Train Perplexity: 72.523
Epoch: 5 Eval Perplexity: 107.383
Test Perplexity: 104.439
"""