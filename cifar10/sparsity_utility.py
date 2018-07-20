from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

class SparsityUtil:
  def __init__(self, block_size):
    self._block_size = block_size

  def zero_block_ratio_matrix(self, a, shape):
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
    n_block_row = int(n_row + self._block_size - 1) / int(self._block_size)
    n_block_col = int(n_col + self._block_size - 1) / int(self._block_size)

    n_blocks = n_block_row * n_block_col

    if n_row % self._block_size != 0:
      n_padded_zeros_in_row = self._block_size - n_block_row % self._block_size
    else:
      n_padded_zeros_in_row = 0
    if n_col % self._block_size != 0:
      n_padded_zeros_in_col = self._block_size - n_block_col % self._block_size
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
    reshaped_a = padded_a.reshape((n_block_row, self._block_size, n_col))
    # Sum the elements within each block column-wise
    summed_a_row = np.sum(reshaped_a, axis=1)
    # Reshape the summed array to a new tensor row-wise
    reshaped_a = summed_a_row.reshape((n_block_row, n_block_col, self._block_size))
    # Sum the elements within each block row-wise
    summed_a = np.sum(reshaped_a, axis=2)
    zero_element_indices = np.where(summed_a == 0)
    zero_counts = len(zero_element_indices[0])

    return float(zero_counts)/float(summed_a.size)
