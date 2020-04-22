# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/7 -*-

import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 持久化存储
v1 = tf.Variable(tf.constant(value=1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(value=2.0, shape=[1]), name='v2')
result = v1 + v2

init_op = tf.global_variables_initializer()
# 声明tf.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess=sess, save_path='./model/model.ckpt')

# 加载已保存的模型
v1 = tf.Variable(tf.constant(value=1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(value=2.0, shape=[1]), name='v2')
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess=sess, save_path='./model/model.ckpt')
    print(sess.run(result))

# 直接加载持久化的图
saver = tf.train.import_meta_graph(meta_graph_or_file='./model/model.ckpt.meta')
with tf.Session() as sess:
    saver.restore(sess=sess, save_path='./model/model.ckpt')
    # 通过张量的名称来获取张量
    print(sess.run(fetches=tf.get_default_graph().get_tensor_by_name(name='add:0')))

# 保存滑动平均模型
v = tf.Variable(0, dtype=tf.float32, name='v')
# 未声明滑动平均模型时只有一个变量v，所以一下语句只会输出‘v:0’
for variables in tf.global_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(decay=0.99)
maintain_averages_op = ema.apply(tf.global_variables())
# 在声明滑动平均模型之后，TensorFlow会自动生成一个影子变量v/ExponentialMoving Average
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(ref=v, value=10))
    sess.run(maintain_averages_op)
    saver.save(sess=sess, save_path='./model/model.ckpt')
    print(sess.run([v, ema.average(v)]))

v = tf.Variable(0, dtype=tf.float32, name='v')
# 通过变量重命名将原来变量v的滑动平均值直接赋值给v
saver = tf.train.Saver({'v/ExponentialMovingAverage': v})
with tf.Session() as sess:
    saver.restore(sess=sess, save_path='./model/model.ckpt')
    print(sess.run(v))  # 输出滑动平均值0.099999905

# 使用tf.variables_to_restore函数可便捷加载重命名滑动平均变量
v = tf.Variable(0, dtype=tf.float32, name='v')
ema = tf.train.ExponentialMovingAverage(decay=0.99)
print(ema.variables_to_restore())
# {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
saver = tf.train.Saver(ema.variables_to_restore())  # ema.variables_to_restore()就相当于上面指定的重命名字典
with tf.Session() as sess:
    saver.restore(sess=sess, save_path='./model/model.ckpt')
    print(sess.run(v))  # 输出滑动平均值0.099999905

# tf.train.Saver会保留全部会话，有时我们只需要模型的计算图
v1 = tf.Variable(tf.constant(value=1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(value=2.0, shape=[1]), name='v2')
result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    # 将图中的变量及其取值转换为常量(tf.gfile括文件的读取、写入、删除、复制等)
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    with tf.gfile.GFile(name='./model/combined_model.pb', mode='wb') as f:  # pb文件为一种模型保存的格式
        f.write(file_content=output_graph_def.SerializeToString())

# 当只需要得到计算图中某个节点的取值时，可以进行如下操作
with tf.Session() as sess:
    model_filename = './model/combined_model.pb'
    # 读取保存的模型文件，并将文件解析成对应的Graph Protocol Buffer(结构化数据处理工具)
    with gfile.FastGFile(name=model_filename, mode='rb') as f:  # 以wb形式写入则以rb形式读取
        graph_def = tf.GraphDef()  # 从Graph中序列化出来的图就叫做GraphDef，图中记录节点的信息，即只有网络的连接信息
        graph_def.ParseFromString(f.read())  # 查看pb文件的节点信息
        print(graph_def)
        # 将graph_def中保存的图加载到当前图中
        result = tf.import_graph_def(graph_def=graph_def, return_elements=['add:0'])
        print(sess.run(result))

"""
node {
  name: "v1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "v1/read"
  op: "Identity"
  input: "v1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@v1"
      }
    }
  }
}
node {
  name: "v2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "v2/read"
  op: "Identity"
  input: "v2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@v2"
      }
    }
  }
}
node {
  name: "add"
  op: "Add"
  input: "v1/read"
  input: "v2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
library {
}

[array([3.], dtype=float32)]
"""
