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

"""Builds the ResNet-50 network.
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

import imagenet_input
import sparsity_util

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/imagenet/tf/train',
                           """Path to the ResNet-50 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')

# Global constants describing the ResNet-50 data set.
NUM_CLASSES = 1001
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1281167
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 50000


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.94  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.005       # Initial learning rate.
WEIGHT_DECAY = 0.004

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# THESE FUNCTIONS ARE SPECIFIC TO RESNET

BN_EPSILON = 0.001

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), weight_decay=WEIGHT_DECAY, is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)

    if weight_decay is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(new_variables), weight_decay, name='weight_loss')
        tf.compat.v1.add_to_collection('losses', weight_loss)
    return new_variables


def output_layer(input_layer, num_labels,monitored_tensor_list):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h, monitored_tensor_list


def batch_normalization_layer(input_layer, dimension,monitored_tensor_list):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer, monitored_tensor_list

# This is only used for the first layer
def conv_bn_relu_layer(input_layer, filter_shape, stride, monitored_tensor_list):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, 
                              strides=[1, stride, stride, 1], padding='SAME')
    bn_layer, monitored_tensor_list = batch_normalization_layer(conv_layer, 
            out_channel, monitored_tensor_list=monitored_tensor_list)

    output = tf.nn.relu(bn_layer)
    
    return output, monitored_tensor_list

# This is the main block and is used on the rest of the network
def bn_relu_conv_layer(input_layer, filter_shape, stride, monitored_tensor_list):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer, monitored_tensor_list = batch_normalization_layer(input_layer, in_channel, monitored_tensor_list=monitored_tensor_list)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    
    filter_size = filter.get_shape().as_list()[0]
    scope_name = tf.get_default_graph().get_name_scope()
    if 'rb' in scope_name:
      out_channel_size = filter.get_shape().as_list()[3]
      num_batches_needed = 4
      if 'rb2' in scope_name:
        num_batches_needed = 16
      elif 'rb3' in scope_name or 'rb4' in scope_name:
        num_batches_needed = 64
      sliced_relu = tf.slice(relu_layer, [0, 0, 0, 0], [num_batches_needed, relu_layer.shape[1], relu_layer.shape[2], relu_layer.shape[3]])
      im2col_relu = tf.extract_image_patches(sliced_relu,
                                           [1,filter_size, filter_size, 1],
                                           [1, stride, stride, 1], [1, 1, 1, 1],
                                           padding='SAME', name='im2col')
      monitored_tensor_list.append(im2col_relu)
    #monitored_tensor_list.append(relu_layer)
    return conv_layer, monitored_tensor_list



def residual_block(input_layer, output_channel, monitored_tensor_list=[]):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1'):
        filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
        conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, stride, stride, 1], padding='SAME')

    with tf.variable_scope('conv2'):
        conv2 , monitored_tensor_list= bn_relu_conv_layer(conv1, 
                [3, 3, output_channel, output_channel], 1, 
                monitored_tensor_list)

    conv2_bn, monitored_tensor_list = batch_normalization_layer(conv2, output_channel, monitored_tensor_list=monitored_tensor_list) 

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], 
                                            [input_channel // 2,
                                            input_channel // 2]])
    else:
        padded_input = input_layer

    pre_activation = conv2_bn + padded_input
    output = tf.nn.relu(pre_activation)
    scope_name = tf.get_default_graph().get_name_scope()
    if 'rb' in scope_name:
      num_batches_needed = 4
      if 'rb2' in scope_name:
        num_batches_needed = 16
      elif 'rb3' in scope_name or 'rb4' in scope_name:
        num_batches_needed = 64
      sliced_output = tf.slice(output, [0, 0, 0, 0], [num_batches_needed, output.shape[1], output.shape[2], output.shape[3]])
      im2col_output = tf.extract_image_patches(sliced_output,
                                       [1, 3, 3, 1],
                                       [1, 1, 1, 1], [1, 1, 1, 1],
                                       padding='SAME', name='im2col')
      monitored_tensor_list.append(im2col_output)
    #monitored_tensor_list.append(output)

    return output, monitored_tensor_list

def inference(input_tensor_batch, n=[2, 2, 2, 2], reuse=False):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    monitored_tensor_list = []
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0, monitored_tensor_list = conv_bn_relu_layer(
                input_tensor_batch, [7, 7, 3, 64], 2,
                monitored_tensor_list)
        layers.append(conv0)

    # pool1
    pool1 = tf.nn.max_pool2d(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    layers.append(pool1)
    num_batches_needed = 4
    sliced_pool1 = tf.slice(pool1, [0, 0, 0, 0], [num_batches_needed, pool1.shape[1], pool1.shape[2], pool1.shape[3]])
    im2col_pool1 = tf.extract_image_patches(sliced_pool1,
                                 [1, 3, 3, 1],
                                 [1, 1, 1, 1], [1, 1, 1, 1],
                                 padding='SAME', name='pool1')
    monitored_tensor_list.append(im2col_pool1)
    #monitored_tensor_list.append(pool1)

    # First residual block set
    for i in range(n[0]):
        with tf.variable_scope('rb1_%d' %i, reuse=reuse):
            conv1, monitored_tensor_list = residual_block(layers[-1], 64,
                    monitored_tensor_list=monitored_tensor_list)
            layers.append(conv1)

    # First residual block set
    for i in range(n[1]):
        with tf.variable_scope('rb2_%d' %i, reuse=reuse):
            conv2, monitored_tensor_list = residual_block(layers[-1], 128,
                    monitored_tensor_list=monitored_tensor_list)
            layers.append(conv2)

    # First residual block set
    for i in range(n[2]):
        with tf.variable_scope('rb3_%d' %i, reuse=reuse):
            conv3, monitored_tensor_list = residual_block(layers[-1], 256,
                    monitored_tensor_list=monitored_tensor_list)
            layers.append(conv3)
    
    # First residual block set
    for i in range(n[3]):
        with tf.variable_scope('rb4_%d' %i, reuse=reuse):
            conv4, monitored_tensor_list = residual_block(layers[-1], 512,
                    monitored_tensor_list=monitored_tensor_list)
            layers.append(conv4)

    # If you are interested on going to 512 channels, you must copy the last
    # block and adjust channel size it accordingly.
    with tf.variable_scope('fc', reuse=reuse):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [512] # Match Nchannels
        output, monitored_tensor_list = output_layer(global_pool, NUM_CLASSES,
                monitored_tensor_list)
        layers.append(output) 
    return layers[-1], monitored_tensor_list

def distorted_inputs():
  """Construct distorted input for ResNet training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  images, labels = imagenet_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                                  num_batches=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

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
  """Add summaries for losses in ResNet-50 model.
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
  """Train ResNet-50 model.
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
  #grad_retrieve_list = sparsity_util.sparsity_hook_backward(total_loss, tensor_list)
  grad_retrieve_list = []
  retrieve_list = retrieve_list + grad_retrieve_list
  print("LOG: Retrieved lists:")
  for t in retrieve_list:
    print (t[0].op.name)

  print("LOG: ---")
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
    opt = tf.compat.v1.train.GradientDescentOptimizer(lr)
    opt = tf.contrib.estimator.clip_gradients_by_norm(
                opt, 5)
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

      #if(len(list(grad.get_shape()))==4):
      #  #temp_grad = tf.nn.relu(tf.sign(tf.abs(grad)))*255
      #  #temp_grad = tf.nn.relu(tf.sign(grad))*255
      #  temp_grad = tf.abs(grad)
      #  print("bingo", temp_grad[:,:,:,0:1].get_shape())
      #  tf.summary.image(var.op.name + '/gradients/img-tmp', temp_grad[:,:,:,0:1])
      #  tf.summary.image(var.op.name + '/gradients/img-grad', (tf.nn.relu(tf.sign(tf.abs(grad)))*255)[:,:,:,0:1])

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op, retrieve_list



