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

"""Builds the CIFAR-10 network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import collections

from six.moves import urllib
import tensorflow as tf

import cifar10_input
import sparsity_util

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def inference(images):
    """Build the VGG-11 model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
    Returns:
        Logits.
    """
    monitored_tensor_list = []
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv 1.1
    with tf.variable_scope('conv1_1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(pre_activation, name="relu")
        #monitored_tensor_list.append(conv1_1)

    # pool1
    pool1 = tf.nn.max_pool(conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    #monitored_tensor_list.append(pool1)
    
    # conv 2.1
    with tf.variable_scope('conv2_1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(pre_activation, name="relu")
        #monitored_tensor_list.append(conv2_1)

    # pool2
    pool2 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    #monitored_tensor_list.append(pool2)


    # conv 3.1
    with tf.variable_scope('conv3_1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(pre_activation, name="relu")
        monitored_tensor_list.append(conv3_1)

    # conv 3.2
    with tf.variable_scope('conv3_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(pre_activation, name="relu")
        #monitored_tensor_list.append(conv3_2)

    # pool3
    pool3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    #monitored_tensor_list.append(pool3)

    # conv 4.1
    with tf.variable_scope('conv4_1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 512], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(pre_activation, name="relu")
        monitored_tensor_list.append(conv4_1)

    # conv 4.2
    with tf.variable_scope('conv4_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(pre_activation, name="relu")
        #monitored_tensor_list.append(conv4_2)

    # pool4
    pool4 = tf.nn.max_pool(conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    #monitored_tensor_list.append(pool4)

    # conv 5.1
    with tf.variable_scope('conv5_1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(pre_activation, name="relu")
        #im2col_conv51data = tf.extract_image_patches(conv5_1,
        #                                 [1, 3, 3, 1],
        #                                 [1, 1, 1, 1], [1, 1, 1, 1],
        #                                 padding='SAME')
        monitored_tensor_list.append(conv5_1)

    # conv 5.2
    with tf.variable_scope('conv5_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(pre_activation, name="relu")
        #monitored_tensor_list.append(conv5_2)

    # pool5
    pool5 = tf.nn.max_pool(conv5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    #monitored_tensor_list.append(pool5)

    # dense1
    with tf.variable_scope('dense1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool5, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
        dense1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="relu")
        #monitored_tensor_list.append(dense1)
        

    # dense2
    with tf.variable_scope('dense2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4096, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
        dense2 = tf.nn.relu(tf.matmul(dense1, weights) + biases, name="relu")
        #monitored_tensor_list.append(dense2)

    # dense3
    with tf.variable_scope('softmax') as scope:
        weights = _variable_with_weight_decay('weights', [4096, NUM_CLASSES], stddev=1/4096.0, wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax = tf.add(tf.matmul(dense2, weights), biases, name="output")

    return softmax, monitored_tensor_list


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, tensor_list, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """

  retrieve_list = sparsity_util.sparsity_hook_forward(tensor_list)
  # grad_retrieve_list = sparsity_util.sparsity_hook_backward(total_loss, tensor_list)
  grad_retrieve_list = []
  retrieve_list = retrieve_list + grad_retrieve_list
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add sparsity for trainable variables.
  #for var in tf.trainable_variables():
  #  tf.summary.scalar(var.op.name, tf.nn.zero_fraction(var))

  # Add sparsity for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.scalar(var.op.name + '/gradients/sparsity', tf.nn.zero_fraction(grad))

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op, retrieve_list


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
