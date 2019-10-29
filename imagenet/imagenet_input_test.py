from __future__ import print_function

import argparse
import os
import threading
import time

import numpy as np
from datetime import datetime

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest
import datasets
import imagenet_input

import numpy as np

data_dir = '/spartan/tf/train'
num_batches = 16

cpu_device = '/cpu:0'

with tf.Graph().as_default() as g:                                            
  global_step = tf.compat.v1.train.get_or_create_global_step()

  with tf.device('/cpu:0'):
    images, labels = imagenet_input.distorted_inputs(data_dir, num_batches)
  
  summary_op = tf.compat.v1.summary.merge_all()
  
  # Sets up a timestamped log directory.
  #logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d")
  #summary_writer = tf.compat.v1.summary.FileWriter(logdir, g)
  
  with tf.compat.v1.Session() as sess:
    final_batches = sess.run(images)
    final_labels = sess.run(labels)
    print(final_batches)
    print(final_labels)
  

