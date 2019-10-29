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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest
import datasets
import numpy as np

# Process images of this size. Note that this differs from the ImageNet data
# image size of 224 x 224. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 224

def get_image_preprocessor(dataset, image_size, num_batches, num_gpus, input_data_type,
                           shift_ratio):
  """Returns the image preprocessor to used, based on the model.

  Returns:
    The image preprocessor, or None if synthetic data should be used.
  """

  processor_class = dataset.get_image_preprocessor()
  if processor_class is not None:
    return processor_class(
        image_size, image_size, num_batches,
        num_gpus, dtype=input_data_type, train=True,
        distortions=False, resize_method='bilinear',
        shift_ratio=shift_ratio)
  else:
    assert isinstance(dataset, datasets.SyntheticData)
    return None

def add_image_preprocessing(dataset, image_preprocessor, input_nchan,
                            image_size, batch_size, num_compute_devices,
                            input_data_type, train):
  """Add image Preprocessing ops to tf graph."""
  nclass = dataset.num_classes() + 1
  if train:
    subset = 'train'
  else:
    subset = 'validation'
  if image_preprocessor is not None:
    images, labels = image_preprocessor.minibatch(
        dataset, subset=subset, use_data_sets=False)
    images_splits = images
    labels_splits = labels
  else:
    assert isinstance(dataset, datasets.SyntheticData)
    input_shape = [batch_size, image_size, image_size, input_nchan]
    images = tf.truncated_normal(
        input_shape,
        dtype=input_data_type,
        stddev=1e-1,
        name='synthetic_images')
    labels = tf.random_uniform(
        [batch_size],
        minval=1,
        maxval=nclass,
        dtype=tf.int32,
        name='synthetic_labels')
    # Note: This results in a H2D copy, but no computation
    # Note: This avoids recomputation of the random values, but still
    #         results in a H2D copy.
    images = tf.contrib.framework.local_variable(images, name='images')
    labels = tf.contrib.framework.local_variable(labels, name='labels')
    # Change to 0-based (don't use background class like Inception does)
    labels -= 1
    if num_compute_devices == 1:
      images_splits = [images]
      labels_splits = [labels]
    else:
      images_splits = tf.split(images, num_compute_devices, 0)
      labels_splits = tf.split(labels, num_compute_devices, 0)
  return nclass, images_splits, labels_splits


def distorted_inputs(data_dir, num_batches):

  # input format.
  dataset = datasets.create_dataset(data_dir, 'imagenet')
  image_size = 224
  num_gpus = 1
  input_nchan = 3
  input_data_type = tf.float32
  shift_ratio = 0
  device_idx = num_gpus - 1

  tf.compat.v1.set_random_seed(1234)
  np.random.seed(4321)
  phase_train = True
  image_preprocessor = get_image_preprocessor(dataset, image_size, num_batches, num_gpus,
                                              input_data_type, shift_ratio)

  nclass, images_splits, labels_splits = add_image_preprocessing(
    dataset, image_preprocessor, input_nchan, image_size,
    num_batches, 1, input_data_type, True)

  return images_splits[device_idx], labels_splits[device_idx]

