# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/3/29 -*-

# Session（会话）来执行定义好的运算
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([[1, 2], [3, 4]], name='a', dtype=tf.float32)
print(a)

# 用上下文资源管理器，不管程序有无异常，最后都可以进行资源回收，不会产生资源泄露的问题。
# 所以只要把所有的计算都放在“with”的内部就行了。
# （1）计算张量的取值
with tf.Session() as sess:
    print(sess.run(a))

# （2）计算张量的取值：设定默认会话来获取张量的取值
sess = tf.Session()
with sess.as_default():  # 注册为默认会话
    print(a.eval())

# （3）在jupyter等交互环境里，使用InteractiveSession可以直接生成默认会话，无需用with语句声明
sess = tf.InteractiveSession()
print(a.eval())
sess.close()

# （4）通过ConfigProto配置会话
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)