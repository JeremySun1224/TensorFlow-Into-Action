# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/3/29 -*-

# 从功能角度上，张量可以被简单理解为多维数组
# 在张量中并没有真正的保存数字，它保存的是如何得到这些数字的计算过程
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.constant是一个计算，这个计算的结果为一个张量，保存在变量a中
a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')
c = tf.constant([1, 2], name='c')

result = tf.add(a, b, name='add')
print(result)

with tf.Session() as sess:
    result = sess.run(result)
    print(result)

print(a)  # dtype=float32，float表示实数
print(c)  # dtype=int32，int表示整数
# dtype类型不同不能相加

# 所以一般建议在声明变量的时候加上dtype
d = tf.constant([1, 2], name='d', dtype=tf.float32)
print(d)