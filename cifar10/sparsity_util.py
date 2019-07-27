from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

TOWER_NAME = 'tower'

def get_dim_sparsity(input_tensor,dim=0):
    dim_size = input_tensor.shape[dim]
    dim_nz=tf.count_nonzero(input_tensor, dim)
    dim_density = dim_nz/dim_size
    sparsity_tensor = 1-dim_density
    return sparsity_tensor

def get_image_patches(input_tensor, filter_size):
    image_patches = tf.extract_image_patches(input_tensor,
                                         [1,filter_size, filter_size, 1],
                                         [1, 1, 1, 1], [1, 1, 1, 1],
                                         padding='SAME')
    return image_patches

def sparsity_hook_forward(x_list):
  """Helper to create summaries of sparsity.

  Creates a summary that measures the sparsity of a tensor.

  Args:
    x_list: a list of tensor
  Returns:
    tensor and sparsity tuble
  """

  retrieve_list = []
  for x in x_list:
    tensor_name = x.op.name
    # print('Forward - ' + x.op.name)
    # print(x.shape)
    # print()
    
    # Get regular sparsity
    sparsity = tf.nn.zero_fraction(x)
    #if ("conv" in tensor_name):
    #    print(x.op.name + ' ' + x.shape)
    tf.summary.scalar(tensor_name + '/sparsity', sparsity)
    retrieve_list.append((x, sparsity))

    # get column sparsity
    # patches differ from model to model
    if FLAGS.network_type == 'cifar10':
        im2col=get_image_patches(x,5)
        col_sparsity = get_dim_sparsity(im2col,3)
        tf.summary.histogram(tensor_name + '/sparsity_histo',col_sparsity)

    if FLAGS.network_type == 'resnet50':
        if 'block/Relu' in tensor_name:
            im2col=get_image_patches(x,3)
            print ("FOR block/Relu im2col shape:")
            print (im2col.shape)
            print ()
            col_sparsity = get_dim_sparsity(im2col,3)
            tf.summary.histogram(tensor_name + '/sparsity_histo',col_sparsity)

    retrieve_list.append((x, sparsity))

  return retrieve_list

def sparsity_hook_backward(loss, x_list):
  """Helper to create summaries for gradients of intermediate results in
  backward pass.

  Creates a summary that measures the sparsity of gradients of intermediate
  results in backward pass.

  Args:
    loss: the loss
    x_list: a list of Tensors
  Returns:
    a list of tensor and sparsity tuple
  """
  gradient_list = tf.gradients(loss, x_list)
  grad_retrieve_list = []
  for g in gradient_list:
    tensor_name = g.op.name
    # print('Back - ' + g.op.name)
    # print(g.shape)
    # print()

    # Get full sparsity
    sparsity = tf.nn.zero_fraction(g)
    tf.summary.scalar(tensor_name + '/sparsity', sparsity)
    grad_retrieve_list.append((g, sparsity))
    
    if FLAGS.network_type == 'resnet50':
        if 'gradients/AddN' in tensor_name:
            im2col=get_image_patches(g,3)
            print ("BACK gradients/AddN im2col shape:")
            print (im2col.shape)
            print ()
            col_sparsity = get_dim_sparsity(im2col,3)
            tf.summary.histogram(tensor_name + '/sparsity_histo',col_sparsity)
  return grad_retrieve_list

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

def feature_map_extraction(tensor, data_format, batch_index, channel_index):
  # The feature map returned will be represented in a context of matrix
  # sparsity (1 or 0), in which 1 means non-zero value, 0 means zero
  n_dim = len(tensor.shape)
  if n_dim == 4:
    if data_format == "NCHW":
      extracted_subarray = tensor[batch_index,channel_index,:,:]
    elif data_format == "NHWC":
      extracted_subarray = tensor[batch_index,:,:,channel_index]
  if n_dim == 2:
    extracted_subarray = tensor
  extracted_subarray[np.nonzero(extracted_subarray)] = 1
  return extracted_subarray

def zero_block_ratio_matrix(a, shape, block_size):
  '''
  Args:
    a: a numpy n-d array (tensor)
    shape: the tensor shape
  Return:
    The count of zero blocks
  '''
  n_dim = len(shape)
  if n_dim == 2:
    n_row = shape[0].value
    n_col = shape[1].value
    matrix = a
  elif n_dim == 4:
    n_row = shape[0].value
    n_col = shape[1].value * shape[2].value * shape[3].value
    matrix = a.reshape((n_row, n_col))
  n_block_row = int(n_row + block_size - 1) / int(block_size)
  n_block_col = int(n_col + block_size - 1) / int(block_size)

  n_blocks = n_block_row * n_block_col

  if n_row % block_size != 0:
    n_padded_zeros_in_row = block_size - n_block_row % block_size
  else:
    n_padded_zeros_in_row = 0
  if n_col % block_size != 0:
    n_padded_zeros_in_col = block_size - n_block_col % block_size
  else:
    n_padded_zeros_in_col = 0

  if n_padded_zeros_in_row != 0 or n_padded_zeros_in_col != 0:
    padded_zeros_in_row = np.zeros((n_padded_zeros_in_row, n_col))
    padded_zeros_in_col = np.zeros((n_row+n_padded_zeros_in_row, n_padded_zeros_in_col))
    padded_a = np.concatenate((np.concatenate((matrix, padded_zeros_in_row), axis=0),\
                padded_zeros_in_col), axis=1)
  else:
    padded_a = matrix

  # Reshape the tensor column-wise first
  reshaped_a = padded_a.reshape((n_block_row, block_size, n_col))
  # Sum the elements within each block column-wise
  summed_a_row = np.sum(reshaped_a, axis=1)
  # Reshape the summed array to a new tensor row-wise
  reshaped_a = summed_a_row.reshape((n_block_row, n_block_col, block_size))
  # Sum the elements within each block row-wise
  summed_a = np.sum(reshaped_a, axis=2)
  zero_element_indices = np.where(summed_a == 0)
  zero_counts = len(zero_element_indices[0])

  return float(zero_counts)/float(summed_a.size)
