# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm

import tensorflow as tf

import cifar10
import block_sparsity_util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_string('sparsity_dir', '/tmp/cifar10_sparsity',
                           """Directory where to write summaries""")
tf.app.flags.DEFINE_integer('monitor_interval', 10,
                           """The interval of monitoring sparsity""")
tf.app.flags.DEFINE_float('sparsity_threshold', 0.6,
                           """The threshold of sparsity to enable monitoring""")
tf.app.flags.DEFINE_boolean('log_animation', False,
                           """Weather or not log the animation for tracking
                           the change of spatial pattern""")
tf.app.flags.DEFINE_integer('batch_idx', 0,
                           """The batch index to extract the feature map""")
tf.app.flags.DEFINE_integer('block_size', 4,
                           """The block size of block-sparse representation""")

def get_non_zero_index(a, shape):
  raw_index = np.where(a != 0)
  n_dim = len(raw_index)
  assert n_dim == 4 or n_dim == 2
  n_data = len(raw_index[0])
  index_list = []
  if n_dim == 4:
    size_chw = shape[1].value * shape[2].value * shape[3].value
    size_hw = shape[2].value * shape[3].value
    size_w = shape[3].value
  elif n_dim == 2:
    size_c = shape[1].value
  for i in range(n_data):
    if n_dim == 4:
      index = raw_index[0][i] * size_chw + raw_index[1][i] * size_hw + raw_index[2][i] * size_w + raw_index[3][i]
    elif n_dim == 2:
      index = raw_index[0][i] * size_c + raw_index[1][i]
    index_list.append(index)
  return index_list

def calc_index_diff_percentage(index_list, ref_index_list, sparsity, all_counts):
  percentage = 1.0
  n_idx = float(len(index_list))
  n_ref_idx = float(len(ref_index_list))
  #print("Current non-zero data size: ", len(index_list))
  #print("Previous non-zero data size: ", len(ref_index_list))
  all_index = np.concatenate((index_list, ref_index_list), axis=0)
  #print("Merged non-zero data size: ", len(all_index))
  #print("Unique non-zero data size: ", len(np.unique(all_index, axis=0)))
  unchanged_counts = len(all_index) - len(np.unique(all_index, axis=0))
  diff_counts = (n_idx - unchanged_counts) + (n_ref_idx - unchanged_counts)
  #print("Differenct counts: ", diff_counts)
  percentage = float(diff_counts) / all_counts
  return percentage

def feature_map_extraction(tensor, batch_index, channel_index):
  # The feature map returned will be represented in a context of matrix
  # sparsity (1 or 0), in which 1 means non-zero value, 0 means zero
  n_dim = len(tensor.shape)
  if n_dim == 4:
    extracted_subarray = tensor[batch_index,:,:,channel_index]
  if n_dim == 2:
    extracted_subarray = tensor
  extracted_subarray[np.nonzero(extracted_subarray)] = 1
  return extracted_subarray

# Store the data for visualization
data_dict = collections.OrderedDict()
# Block: zero Red: non-zero
cmap = ListedColormap(['black', 'red'])

def animate(i, ax, tensor_name):
  global data_dict
  label = 'Local step in monitoring period: {0}'.format(i)
  matrix = data_dict[tensor_name][i]
  mesh = ax.pcolormesh(matrix, cmap=cmap)
  ax.set_xlabel(label)
  return mesh,


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    tensor_list, sparsities = cifar10.inference(images)
    logits = tensor_list[0]

    #Build a new tensor list with both tensor and sparsities
    retrieve_list = []
    for key in sparsities:
      retrieve_list.append(key)
      retrieve_list.append(sparsities[key])

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op, grad_retrieve_list = cifar10.train(loss, tensor_list, global_step)
    retrieve_list = retrieve_list + grad_retrieve_list

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    class _SparsityHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
       self._bs_util = block_sparsity_util.BlockSparsityUtil(FLAGS.block_size)
       self._internal_index_keeper = collections.OrderedDict()
       self._local_step = collections.OrderedDict()
       #self._fig, self._ax = plt.subplots()

      def before_run(self, run_context):
        return tf.train.SessionRunArgs(retrieve_list)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        self._data_list = []
        self._sparsity_list = []
        for i in range(len(run_values.results)):
          if i % 2 == 0:
            # tensor
            self._data_list.append(run_values.results[i])
          if i % 2 == 1:
            # sparsity
            self._sparsity_list.append(run_values.results[i])
        assert len(self._sparsity_list) == len(retrieve_list) / 2
        assert len(self._data_list) == len(retrieve_list) / 2
        num_data = len(self._data_list)
        format_str = ('local_step: %d %s: sparsity = %.2f difference percentage = %.2f')
        zero_block_format_str = ('local_step: %d %s: zero block ratio = %.2f')
        for i in range(num_data):
          sparsity = self._sparsity_list[i]
          shape = retrieve_list[2*i].get_shape()
          tensor_name = retrieve_list[2*i].name
          batch_idx = FLAGS.batch_idx
          channel_idx = int(shape[-1]) - 1
          if tensor_name in self._local_step:
            if self._local_step[tensor_name] == FLAGS.monitor_interval and \
               FLAGS.log_animation:
              fig, ax = plt.subplots()
              ani = animation.FuncAnimation(fig, animate, frames=FLAGS.monitor_interval,
                                            fargs=(ax, tensor_name,),
                                            interval=500, repeat=False, blit=True)                        
              
              figure_name = tensor_name.replace('/', '_').replace(':', '_')
              ani.save(figure_name+'.gif', dpi=80, writer='imagemagick')
              self._local_step[tensor_name] += 1
              continue
            if self._local_step[tensor_name] >= FLAGS.monitor_interval:
              continue
          if tensor_name not in self._local_step and sparsity > FLAGS.sparsity_threshold:
            self._local_step[tensor_name] = 0
            zero_block_ratio = self._bs_util.zero_block_ratio_matrix(self._data_list[i], shape)
            print(zero_block_format_str % (self._local_step[tensor_name], tensor_name, zero_block_ratio))
            print (format_str % (self._local_step[tensor_name], tensor_name,
                                 sparsity, 0.0))
            self._internal_index_keeper[tensor_name] = get_non_zero_index(self._data_list[i], shape)
            if tensor_name not in data_dict:
              data_dict[tensor_name] = []
            data_dict[tensor_name].append(feature_map_extraction(self._data_list[i], batch_idx, channel_idx))
            self._local_step[tensor_name] += 1
          elif tensor_name in self._local_step and self._local_step[tensor_name] > 0:
            # Inside the monitoring interval
            zero_block_ratio = self._bs_util.zero_block_ratio_matrix(self._data_list[i], shape)
            print(zero_block_format_str % (self._local_step[tensor_name], tensor_name, zero_block_ratio))
            data_length = self._data_list[i].size
            local_index_list = get_non_zero_index(self._data_list[i], shape)
            diff_percentage = calc_index_diff_percentage(local_index_list,
              self._internal_index_keeper[tensor_name], sparsity, data_length)
            self._internal_index_keeper[tensor_name] = local_index_list
            print (format_str % (self._local_step[tensor_name], tensor_name,
                                 sparsity, diff_percentage))
            data_dict[tensor_name].append(feature_map_extraction(self._data_list[i], batch_idx, channel_idx))
            self._local_step[tensor_name] += 1
          else:
            continue

    sparsity_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.sparsity_dir, g)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               #tf.train.SummarySaverHook(save_steps=FLAGS.log_frequency, summary_writer=summary_writer, summary_op=sparsity_summary_op),
               _LoggerHook(),
               _SparsityHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  if tf.gfile.Exists(FLAGS.sparsity_dir):
    tf.gfile.DeleteRecursively(FLAGS.sparsity_dir)
  tf.gfile.MakeDirs(FLAGS.sparsity_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
