# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/24 -*-


import io
import sys
from tqdm import tqdm
import jieba
import jieba.analyse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # 改变标准输出的默认编码

# 英文分词
sourceTxt = './en-zh/train.tags.en-zh.en'
targetTxt = './train.txt.en'

with open(file=sourceTxt, mode='r', encoding='utf-8') as sourceFile, open(file=targetTxt, mode='a+', encoding='utf-8') as targetFile:
    for line in tqdm(sourceFile.readlines()):
        seg = jieba.cut(line.strip(), cut_all=False)
        output = ' '.join(seg)
        targetFile.write(output + '\n')
    print('写入成功')

# 中文分词
sourceTxtZH = './en-zh/train.tags.en-zh.zh'
targetTxtZH = './train.txt.zh'

with open(file=sourceTxtZH, mode='r', encoding='utf-8') as sourceFileZH, open(file=targetTxtZH, mode='a+', encoding='utf-8') as targetFileZH:
    for line in tqdm(sourceFileZH.readlines()):
        seg = jieba.cut(line.strip(), cut_all=False)
        output = ' '.join(seg)
        targetFileZH.write(output + '\n')
    print('写入成功')
