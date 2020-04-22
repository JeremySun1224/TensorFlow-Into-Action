# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/21 -*-

# 为了将文本转化为模型可以读入的单词序列，需要将这10000个不同的词汇分别映射到0~9999之间的整数编号
import codecs
import collections
from operator import itemgetter  # 获取对象指定维度的信息

RAW_DATA = '../data/ptb.train.txt'  # 训练集数据文件
VOCAB_OUTPUT = 'ptb.vocab'  # 输出的词汇表文件

# 统计单词出现频率
counter = collections.Counter()
with codecs.open(filename=RAW_DATA, mode='r', encoding='utf-8') as f:  # codecs.open()不易出现编码问题
    for line in f:
        for word in line.strip().split():
            counter[word] += 1  # 返回一个类似字典，记录了每个word出现的次数，即{word: 次数}
# 按词频顺序对单词进行排序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)  # 翻转，即高频在前
sorted_words = [x[0] for x in sorted_word_to_cnt]
"""
以word = [1, 2, 2, 4, 5, 3, 2, 0, 9]为例：
Counter({2: 3, 1: 1, 4: 1, 5: 1, 3: 1, 0: 1, 9: 1})
dict_items([(1, 1), (2, 3), (4, 1), (5, 1), (3, 1), (0, 1), (9, 1)])
[(2, 3), (1, 1), (4, 1), (5, 1), (3, 1), (0, 1), (9, 1)]
[2, 1, 4, 5, 3, 0, 9]
"""

# 稍后我们需要在文本换行处加入句子结束符"<eos>"，这里预先将其加入词汇表
sorted_words = ['<eos>'] + sorted_words  # 这里的sorted_words就是词汇表
# 在处理机器翻译数据时，我们还需要加入"<unk>"和句子起始符"<sos>"，并删除低频词汇，代码如下：
# sorted_words = ['<unk>', '<sos>', 'eos'] + sorted_words
# if len(sorted_words) > 10000:
#     sorted_words = sorted_words[:10000]

with codecs.open(filename=VOCAB_OUTPUT, mode='w', encoding='utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')
