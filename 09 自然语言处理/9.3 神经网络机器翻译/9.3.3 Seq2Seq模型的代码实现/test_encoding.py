# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/4/25 -*-

import os
from tensorflow.python import pywrap_tensorflow

model_dir = './'
checkpoint_path = os.path.join(model_dir, "seq2seq_ckpt-2000")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
