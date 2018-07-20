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

from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import sparsity_utility

class ModelInfo:
  def __init__(self, global_step):
    self._global_step = global_step

class Mode(Enum):
  monitor = 0
  compute = 1

class Status(Enum):
  hibernate = -1
  active = 1

class SparseStage(Enum):
  dense = 0
  first_stage = 1   # 60 ~ 70%
  second_stage = 2  # 70 ~ 80%
  third_stage = 3   # 80 ~ 90%
  fourth_stage = 4  # 90 ~ 100%
  invalid_stage = 5 # 100%

class MonitoredTensorInfo:
  def __init__(self):
    self._sparsity = -1.0
    self._zero_block_ratio = 0.0
    self._non_zero_data_indices = []
    self._local_step = 0
    self._extracted_data_list = []
    self._fd = None
    self._enabled = False
    self._data_tensor = None
    self._sparsity_tensor = None

    self._previous_sparsity = -1.0
    self._stage = SparseStage.dense
    self._monitor_period_counts = 0

  def reset()
    self._sparsity = -1.0
    self._zero_block_ratio = 0.0
    self._non_zero_data_indices = []
    self._local_step = 0
    self._extracted_data_list = []
    self._fd = None
    self._enabled = False
    self._data_tensor = None
    self._sparsity_tensor = None

class SparsityMonitor:
  def __init__(self, mode, data_fromat,
               monitor_interval, monitor_period,
               retrieved_tensor_list):
    self._can_hibernate = False
    self._mode = mode 
    self._data_format = data_format
    # The interval within which every iteration is used for monitoring sparsity
    self._monitor_interval = monitor_interval
    # The period of initiating one monitoring session
    self._initial_monitor_period = monitor_period
    self._monitor_period = self._initial_monitor_period

    # Fixed parameter during running
    self._monitor_period_incremental = 1000
    self._incremental_counts = 0
    self._incremental_times = 10
    self._monitor_period_multiple = 2
    self._hibernation_period = 10000
    self._hibernation_cond_sparsity_difference = 0.01
    self._hibernation_cond_n_period = 5
    self._initial_sparsity = 0.6
    self._sparsity_threshold_list = np.arange(self._initial_sparsity, 1, 0.1)
    self._first_sparse_stage_idx = 0
    self._second_sparse_stage_idx = 1
    self._third_sparse_stage_idx = 2
    self._forth_sparse_stage_idx = 3

    # Initial status
    self._status = Status.active

    # In the data list, the even index points to monitored data
    # the odd index points to the corresponding sparsity
    self._retrieved_tensor_list = retrieved_tensor_list
    self._monitor_enabled_tensor_list = []
    self._sparsity_info = []
    for tensor_tuple in self._retrieved_tensor_list:
      tensor_name = tensor_tuple[0].name
      self._sparsity_info.append(MonitoredTensorInfo())
      self._sparsity_info[-1]._data_tensor = tensor_tuple[0]
      self._sparsity_info[-1]._sparsity_tensor = tensor_tuple[1]
    self._num_sparsity_info = len(self._sparsity_info)

  def update_stage(self, tensor_idx):
    if self._sparsity_info[tensor_idx]._sparsity >= self._sparsity_threshold_list[self._first_sparse_stage_idx] \
      and self._sparsity_info[tensor_idx]._sparsity < self._sparsity_threshold_list[self._second_sparse_stage_idx]:
      self._sparsity_info[tensor_idx]._stage = SparseStage.first_stage
    elif self._sparsity_info[tensor_idx]._sparsity >= self._sparsity_threshold_list[self._second_sparse_stage_idx] \
      and self._sparsity_info[tensor_idx]._sparsity < self._sparsity_threshold_list[self._third_sparse_stage_idx]:
      self._sparsity_info[tensor_idx]._stage = SparseStage.second_stage
    elif self._sparsity_info[tensor_idx]._sparsity >= self._sparsity_threshold_list[self._third_sparse_stage_idx] \
      and self._sparsity_info[tensor_idx]._sparsity < self._sparsity_threshold_list[self._fourth_sparse_stage_idx]: 
      self._sparsity_info[tensor_idx]._stage = SparseStage.third_stage
    elif self._sparsity_info[tensor_idx]._sparsity >= self._sparsity_threshold_list[self._fourth_sparse_stage_idx] \
      and self._sparsity_info[tensor_idx]._sparsity < 1.0:
      self._sparsity_info[tensor_idx]._stage = SparseStage.fourth_stage
    else:
      self._sparsity_info[tensor_idx]._stage = SparseStage.invalid_stage

  def update_sparsity_history(self, tensor_idx, history_length):

  def adjust_monitor_period(self):
    

  def scheduler_before(self, model_info):
    # Manage the status
    if self._can_hibernate:
      self._status = Status.hibernation
    else:
      self._status = Status.active

    # Schedule the sparsity monitor
    if self._status == Status.active:
      if model_info._global_step % self._monitor_period == 0:
        for i in range(self._num_sparsity_info):
          self._sparsity_info[i]._enabled = True
    elif self._status == Status.hibernate:
      if model_info._global_step % self._hibernation_period == 0:
        for i in range(self._num_sparsity_info):
          self._sparsity_info[i]._enabled = True

    self._monitor_enabled_tensor = []
    for i in range(self._num_sparsity_info):
      if self._sparsity_info[i]._enabled:
        self._monitor_enabled_tensor_list.append(self._sparsity_info[i]._data_tensor)
        self._monitor_enabled_tensor_list.append(self._sparsity_info[i]._sparsity_tensor)

    # Return the data list that needed for monitoring
    return self._monitor_enabled_tensor

  def scheduler_after(value_list, work_path="", enable_file_io=False):
    all_disabled = True
    for i in range(self._num_sparsity_info):
      if !self._sparsity_info[i]_enabled:
        continue
      all_diabled = False
      # Time consuming part is here
      if self._mode == Mode.monitor:
        __collect_and_monitoring(value_list)

    if all_disabled:
      return
    # post processing of scheduler parameters
    if self._status == Status.active:
      for i in range(self._num_sparsity_info):
        if !self._sparsity_info[i]_enabled:
          continue
        # Turn off the monitoring
        if self._sparsity_info[i]._local_count == self._monitor_interval:
          self._sparsity_info[i]._enabled = False
          self._sparsity_info[i]._monitor_period_counts += 1
          # Update the sparsity history based on configured history length
          # Update the sparsity stage of corresponding data
          self.update_stage(i)
          # Adjust the monitor period
          self.adjust_monitor_period()

          # Call file IO utility here TBD

          self._sparsity_info[i].reset()
        elif self._sparsity_info[i]._local_count == 0 and self._sparsity_info[i]._sparsity < 0:
          # The sparsity is too small to be recorded
          self._sparsity_info[i].reset()
          self._sparsity_info[i]._enabled = False

    elif self._status == Status.hibernate:

  def __collect_and_monitoring(value_list, work_path, enable_file_io):
    for i in range(len(value_list)):
      if i % 2 == 0:
        # data 
        self._data_list.append(value_list[i])
      if i % 2 == 1:
        # sparsity
        self._sparsity_list.append(value_list[i])
    assert len(self._sparsity_list) == len(value_list) / 2
    assert len(self._data_list) == len(value_list) / 2
    num_data = len(self._data_list)
    format_str = ('local_step: %d %s: sparsity = %.2f difference percentage = %.2f')
    zero_block_format_str = ('local_step: %d %s: zero block ratio = %.2f')
    for i in range(num_data):
      shape = retrieved_tensor_list[2*i].get_shape()
      tensor = retrieved_tensor_list[2*i]
      self._sparsity_info[tensor]._sparsity = self._sparsity_list[i]
      batch_idx = 0
      channel_idx = 0
      if self._sparsity_info[tensor_name]._local_step == self._monitor_interval and \
         enable_file_io:
        #fig, ax = plt.subplots()
        #ani = animation.FuncAnimation(fig, animate, frames=FLAGS.monitor_interval,
        #                              fargs=(ax, tensor_name,),
        #                              interval=500, repeat=False, blit=True)                        
        #
        #figure_name = tensor_name.replace('/', '_').replace(':', '_')
        #ani.save(figure_name+'.gif', dpi=80, writer='imagemagick')
        self._sparsity_info[tensor_name]._local_step += 1
        continue
      if self._local_step[tensor_name] >= self._monitor_interval:
        continue
      if self._sparsity_info[tensor_name]._local_step == 0 and sparsity > FLAGS.sparsity_threshold:
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
  def __metric_calculator()
    
