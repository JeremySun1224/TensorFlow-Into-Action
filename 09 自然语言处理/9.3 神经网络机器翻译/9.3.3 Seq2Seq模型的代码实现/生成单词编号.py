# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/24 -*-


# 确定了词汇表后，将训练文件、测试文件等根据词汇表转化为单词编号，每个单词的编号就是它在词汇文件中的行号
import codecs

# 英文数据
RAW_DATA_EN = './train.txt.en'
VOCAB_EN = './train.en.vocab'
OUTPUT_DATA_EN = './train.en'

# 读取词汇表并建立词汇到单词编号的映射
with codecs.open(filename=VOCAB_EN, mode='r', encoding='utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


# 如果出现了被删除的低频词，则替换为'<unk>'
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']  # 返回'<unk>'，其编码为2


fin = codecs.open(filename=RAW_DATA_EN, mode='r', encoding='utf-8')
fout = codecs.open(filename=OUTPUT_DATA_EN, mode='w', encoding='utf-8')
for line in fin:
    words = line.strip().split() + ['<eos>']
    # 将每个单词替换为词汇表中的编号
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)
fin.close()
fout.close()

# 中文数据
RAW_DATA_ZH = './train.txt.zh'
VOCAB_ZH = './train.zh.vocab'
OUTPUT_DATA_ZH = './train.zh'

# 读取词汇表并建立词汇到单词编号的映射
with codecs.open(filename=VOCAB_ZH, mode='r', encoding='utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


# 如果出现了被删除的低频词，则替换为'<unk>'
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']  # 返回'<unk>'，其编码为2


fin = codecs.open(filename=RAW_DATA_ZH, mode='r', encoding='utf-8')
fout = codecs.open(filename=OUTPUT_DATA_ZH, mode='w', encoding='utf-8')
for line in fin:
    words = line.strip().split() + ['<eos>']
    # 将每个单词替换为词汇表中的编号
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)
fin.close()
fout.close()
