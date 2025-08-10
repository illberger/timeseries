# data_processor.py
#import numpy as np
import tensorflow as tf
#from line_profiler import LineProfiler, profile

"""
Tensorflow v2.10.1
Numpy v1.23.5
"""

# ClosePrice
mu = tf.constant(69798.43214498, dtype=tf.float32)   # Mean
sigma = tf.constant(12708.4183407934, dtype=tf.float32)  # Std


@tf.function
def preprocess_window(window):
    x = window[:-1, :]
    z_last = window[-2, 3]
    z_next = window[-1, 3]

    raw_last = z_last * sigma + mu
    raw_next = z_next * sigma + mu
    y = (raw_next - raw_last) / raw_last
    return x, tf.expand_dims(y, 0)



