# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/8 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 如何查看保存的变量信息
# tf.train.NewCheckpointReader可以读取checkpoint文件中保存的所有变量，生成的文件名的.data和.index是可以省略的
reader = tf.train.NewCheckpointReader('./model/model.ckpt')

# 获取所有变量列表。这是一个从变量名到变量维度的字典
global_variables = reader.get_variable_to_shape_map()
print(global_variables)
for variable_name in global_variables:
    # 查看变量名称和变量维度
    print(variable_name, global_variables[variable_name])

# 获取名称为v1的变量的取值
print('Value for variable v1 is {value_tensor}'.format(value_tensor=reader.get_tensor('v1')))

"""
{'v1': [1], 'v2': [1]}
v1 [1]
v2 [1]
Value for variable v1 is [1.]
"""
