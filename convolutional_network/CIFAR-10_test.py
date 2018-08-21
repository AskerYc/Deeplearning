import sys
sys.path.append("f:/deeplearning/data_library/models-master/tutorials/image/cifar10")
import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

#parameter define
max_steps = 3000
batch_size = 128
data_dir = 'F:/deeplearning/data_library/cifar10_data/cifar-10-batches-bin'

#init weight
def variable_with_weight_loss(shape, stddev, wl):
	var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
	if wl is not None:
		weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')