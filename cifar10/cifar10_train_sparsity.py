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

import tensorflow as tf

import cifar10

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
tf.app.flags.DEFINE_integer('monitor_interval', 100,
                           """The interval of monitoring sparsity""")
tf.app.flags.DEFINE_float('sparsity_threshold', 0.6,
                           """The threshold of sparsity to enable monitoring""")
def get_non_zero_index(a, shape):
  raw_index = np.where(a != 0)
  n_dim = len(raw_index)
  assert n_dim == 4
  n_data = len(raw_index[0])
  index_list = []
  size_chw = int(shape[1] * shape[2] * shape[3])
  size_hw = int(shape[2] * shape[3])
  size_w = int(shape[3])
  for i in range(n_data):
    index = raw_index[0][i] * size_chw + raw_index[1][i] * size_hw + raw_index[2][i] * size_w + raw_index[3][i]
    index_list.append(index)
  return index_list

def calc_index_diff_percentage(index_list, ref_index_list):
  percentage = 1.0
  n_idx = float(len(index_list))
  all_index = np.concatenate((index_list, ref_index_list), axis=0)
  diff_counts = n_idx - (len(all_index) - len(np.unique(all_index, axis=0)))
  percentage = float(diff_counts) / n_idx
  return percentage

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
       self._internal_data_keeper = collections.OrderedDict()
       self._internal_index_keeper = collections.OrderedDict()
       self._local_step = 0

      def before_run(self, run_context):
        return tf.train.SessionRunArgs(retrieve_list)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        self._data_list = []
        self._sparsity_list = []
        if self._local_step >= FLAGS.monitor_interval:
          return
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
        format_str = ('%s: sparsity = %.2f difference percentage = %.2f')
        for i in range(num_data):
          sparsity = self._sparsity_list[i]
          shape = retrieve_list[2*i].get_shape()
          if self._local_step == 0 and sparsity > FLAGS.sparsity_threshold:
            print (format_str % (retrieve_list[2*i].name,
                                 sparsity, 0.0))
            self._internal_data_keeper[retrieve_list[2*i].name] = self._data_list[i]
            self._internal_index_keeper[retrieve_list[2*i].name] = get_non_zero_index(self._data_list[i], shape)
          elif self._local_step > 0:
            # Inside the monitoring interval
            if retrieve_list[2*i].name not in self._internal_index_keeper:
              continue
            self._internal_data_keeper[retrieve_list[2*i].name] = self._data_list[i]
            local_index_list = get_non_zero_index(self._data_list[i], shape)
            diff_percentage = calc_index_diff_percentage(local_index_list, self._internal_index_keeper[retrieve_list[2*i].name])
            self._internal_index_keeper[retrieve_list[2*i].name] = local_index_list
            print (format_str % (retrieve_list[2*i].name,
                                 sparsity, diff_percentage))
          else:
            continue

        self._local_step += 1           

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
