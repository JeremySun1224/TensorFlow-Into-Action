# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/21 -*-

# 确定了词汇表后，将训练文件、测试文件等根据词汇表转化为单词编号，每个单词的编号就是它在词汇文件中的行号
import codecs
import tensorflow as tf

RAW_DATA = '../../data/ptb.train.txt'
VOCAB = './ptb.vocab'  # 已生成的词汇表文件
OUTPUT_DATA = './ptb.train'  # 将单词替换为单词编号后的输出文件

# 读取词汇表并建立词汇到单词编号的映射
with codecs.open(filename=VOCAB, mode='r', encoding='utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


# 如果出现了被删除的低频词，则替换为'<unk>'
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']  # 返回'<unk>'，其编码为2


fin = codecs.open(filename=RAW_DATA, mode='r', encoding='utf-8')
fout = codecs.open(filename=OUTPUT_DATA, mode='w', encoding='utf-8')
for line in fin:
    words = line.strip().split() + ['<eos>']
    # 将每个单词替换为词汇表中的编号
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)
fin.close()
fout.close()

"""
在输入层，每一个单词用一个实数向量表示，这个向量被称为‘词向量’（word embedding），也可以翻译成‘词嵌入’。
词向量可以形象地理解为将词汇表嵌入到一个固定维度的实数空间里，这样做可以降低输入的维度，并增加语义信息。
假设词向量的维度是EMB_SIZE，词汇表的大小为VOCAB_SIZE，那么所有单词的词向量可以放入一个大小为VOCAB_SIZE*EMB_SIZE的矩阵中。
"""
VOCAB_SIZE = 10000
EMB_SIZE = 128
input_data = OUTPUT_DATA
# 在读取词向量时，可以调用tf.nn.embedding_lookup方法
embedding = tf.get_variable(name='embedding', shape=[VOCAB_SIZE, EMB_SIZE])
# 输出的矩阵会比输入数据多一个维度，新增维度的大小为EMB_SIZE
input_embedding = tf.nn.embedding_lookup(embedding, input_data)
