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
import sparsity_util

import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import os.path
from enum import Enum
import re

tensorname_regex = re.compile(r"(?:.*\/)?([a-z0-9]+)_?\d?:\w+")

problem_size_map = collections.OrderedDict()
problem_size_map['alexnet'] = collections.OrderedDict()
problem_size_map['alexnet']['pool1'] = [0.5, 0.55, 0.6, 0.65]
problem_size_map['alexnet']['pool2'] = [0.45, 0.5, 0.55, 0.6]
problem_size_map['alexnet']['conv3'] = [0.5, 0.55, 0.6]
problem_size_map['alexnet']['conv4'] = [0.5, 0.55, 0.6]

problem_size_map['vggnet'] = collections.OrderedDict()
problem_size_map['vggnet']['conv11'] = [0.5, 0.55, 0.6]
problem_size_map['vggnet']['pool1'] = [0.4, 0.45, 0.5]
problem_size_map['vggnet']['conv21'] = [0.5, 0.55, 0.6]
problem_size_map['vggnet']['pool2'] = [0.35, 0.40, 0.45]
problem_size_map['vggnet']['conv31'] = [0.45, 0.5]
problem_size_map['vggnet']['conv32'] = [0.4, 0.45, 0.5]
problem_size_map['vggnet']['pool3'] = [0.35, 0.40, 0.45]
problem_size_map['vggnet']['conv41'] = [0.4, 0.45, 0.5]
problem_size_map['vggnet']['conv42'] = [0.4, 0.55, 0.5]
problem_size_map['vggnet']['pool4'] = [0.4, 0.45]
problem_size_map['vggnet']['conv51'] = [0.5, 0.55]
problem_size_map['vggnet']['conv52'] = [0.5, 0.55]

problem_size_map['resnet'] = collections.OrderedDict()
problem_size_map['resnet']['pool1_1'] = [0.4, 0.45]
problem_size_map['resnet']['rb1_0/conv2/im2col'] = [0.45, 0.5]
problem_size_map['resnet']['rb1_0/im2col'] = [0.3, 0.35]
problem_size_map['resnet']['rb1_1/conv2/im2col'] = [0.5]
problem_size_map['resnet']['rb1_1/im2col'] = [0.3]
problem_size_map['resnet']['rb2_0/conv2/im2col'] = [0.5]
problem_size_map['resnet']['rb2_0/im2col'] = [0.4]
problem_size_map['resnet']['rb2_1/conv2/im2col'] = [0.5]
problem_size_map['resnet']['rb2_1/im2col'] = [0.3]
problem_size_map['resnet']['rb3_0/conv2/im2col'] = [0.5]
problem_size_map['resnet']['rb3_0/im2col'] = [0.35, 0.4]
problem_size_map['resnet']['rb3_1/conv2/im2col'] = [0.5]
problem_size_map['resnet']['rb3_1/im2col'] = [0.3, 0.35]
problem_size_map['resnet']['rb4_0/conv2/im2col'] = [0.5]
problem_size_map['resnet']['rb4_0/im2col'] = [0.35, 0.4]
problem_size_map['resnet']['rb4_1/conv2/im2col'] = [0.5]
problem_size_map['resnet']['rb4_1/im2col'] = [0.3]

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
    self._zero_block_ratio = -1.0
    self._local_step = 0
    self._extracted_data_list = []
    self._fd = None
    self._results_str = "Monitor session for tensor %s finished: global step: %d\nStatus: %s\nSparsity stage: %s\nMonitor period: %d\nAveraged sparsity = %.2f\nAveraged zero block ratio = %.2f\n"
    self._enabled = False
    self._valid = True

    self._data_tensor = None
    self._sparsity_tensor = None
    self._zero_block_size = 2
    self._sparsity_history = []
    self._stage = SparseStage.dense
    self._monitor_period = 0
    self._real_data_extracted = False
    self._flag_map = collections.OrderedDict()
    self._flag_map['alexnet'] = collections.OrderedDict()
    self._flag_map['alexnet']['pool1'] = [False, False, False, False]
    self._flag_map['alexnet']['pool2'] = [False, False, False, False]
    self._flag_map['alexnet']['conv3'] = [False, False, False]
    self._flag_map['alexnet']['conv4'] = [False, False, False]
    self._flag_map['vggnet'] = collections.OrderedDict()
    self._flag_map['vggnet']['conv11'] = [False, False, False]
    self._flag_map['vggnet']['pool1'] = [False, False, False]
    self._flag_map['vggnet']['conv21'] = [False, False, False]
    self._flag_map['vggnet']['pool2'] = [False, False, False]
    self._flag_map['vggnet']['conv31'] = [False, False]
    self._flag_map['vggnet']['conv32'] = [False, False, False]
    self._flag_map['vggnet']['pool3'] = [False, False, False]
    self._flag_map['vggnet']['conv41'] = [False, False, False]
    self._flag_map['vggnet']['conv42'] = [False, False, False]
    self._flag_map['vggnet']['pool4'] = [False, False]
    self._flag_map['vggnet']['conv51'] = [False, False]
    self._flag_map['vggnet']['conv52'] = [False, False]
    self._flag_map['resnet'] = collections.OrderedDict()
    self._flag_map['resnet']['pool1_1'] = [False, False]
    self._flag_map['resnet']['rb1_0/conv2/im2col'] = [False, False]
    self._flag_map['resnet']['rb1_0/im2col'] = [False, False]
    self._flag_map['resnet']['rb1_1/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb1_1/im2col'] = [False]
    self._flag_map['resnet']['rb2_0/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb2_0/im2col'] = [False]
    self._flag_map['resnet']['rb2_1/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb2_1/im2col'] = [False]
    self._flag_map['resnet']['rb3_0/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb3_0/im2col'] = [False, False]
    self._flag_map['resnet']['rb3_1/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb3_1/im2col'] = [False, False]
    self._flag_map['resnet']['rb4_0/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb4_0/im2col'] = [False, False]
    self._flag_map['resnet']['rb4_1/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb4_1/im2col'] = [False]

  def reset(self):
    self._sparsity = -1.0
    self._zero_block_ratio = -1.0
    self._local_step = 0
    self._extracted_data_list = []
    self._fd = None
    self._results_str = "Monitor session for tensor %s finished: global step: %d\nStatus: %s\nSparsity stage: %s\nMonitor period: %d\nAveraged sparsity = %.2f\nAveraged zero block ratio = %.2f\n"
    self._enabled = False
    self._flag_map = collections.OrderedDict()
    self._flag_map['alexnet'] = collections.OrderedDict()
    self._flag_map['alexnet']['pool1'] = [False, False, False, False]
    self._flag_map['alexnet']['pool2'] = [False, False, False, False]
    self._flag_map['alexnet']['conv3'] = [False, False, False]
    self._flag_map['alexnet']['conv4'] = [False, False, False]
    self._flag_map['vggnet'] = collections.OrderedDict()
    self._flag_map['vggnet']['conv11'] = [False, False, False]
    self._flag_map['vggnet']['pool1'] = [False, False, False]
    self._flag_map['vggnet']['conv21'] = [False, False, False]
    self._flag_map['vggnet']['pool2'] = [False, False, False]
    self._flag_map['vggnet']['conv31'] = [False, False]
    self._flag_map['vggnet']['conv32'] = [False, False, False]
    self._flag_map['vggnet']['pool3'] = [False, False, False]
    self._flag_map['vggnet']['conv41'] = [False, False, False]
    self._flag_map['vggnet']['conv42'] = [False, False, False]
    self._flag_map['vggnet']['pool4'] = [False, False]
    self._flag_map['vggnet']['conv51'] = [False, False]
    self._flag_map['vggnet']['conv52'] = [False, False]
    self._flag_map['resnet'] = collections.OrderedDict()
    self._flag_map['resnet']['pool1_1'] = [False, False]
    self._flag_map['resnet']['rb1_0/conv2/im2col'] = [False, False]
    self._flag_map['resnet']['rb1_0/im2col'] = [False, False]
    self._flag_map['resnet']['rb1_1/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb1_1/im2col'] = [False]
    self._flag_map['resnet']['rb2_0/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb2_0/im2col'] = [False]
    self._flag_map['resnet']['rb2_1/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb2_1/im2col'] = [False]
    self._flag_map['resnet']['rb3_0/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb3_0/im2col'] = [False, False]
    self._flag_map['resnet']['rb3_1/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb3_1/im2col'] = [False, False]
    self._flag_map['resnet']['rb4_0/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb4_0/im2col'] = [False, False]
    self._flag_map['resnet']['rb4_1/conv2/im2col'] = [False]
    self._flag_map['resnet']['rb4_1/im2col'] = [False]

class SparsityMonitor:
  def __init__(self, mode, data_format,
               monitor_interval, initial_monitor_period,
               retrieved_tensor_list):
    self._can_hibernate = False
    self._mode = mode 
    self._data_format = data_format
    # The interval within which every iteration is used for monitoring sparsity
    self._monitor_interval = monitor_interval
    # The period of initiating one monitoring session
    self._initial_monitor_period = initial_monitor_period

    # Fixed parameter during running
    self._monitor_period_multiple = 2
    self._monitor_sparsity_history_length = 10
    self._monitor_cond_sparsity_difference = 0.05
    self._hibernation_period = 10000
    self._hibernation_sparsity_history_length = 3
    self._hibernation_cond_sparsity_difference = 0.1
    self._hibernation_sparsity_history_length = 10
    self._initial_sparsity = 0.3
    self._sparsity_threshold_list = np.arange(self._initial_sparsity, 1, 0.1)
    self._first_sparse_stage_idx = 0
    self._second_sparse_stage_idx = 1
    self._third_sparse_stage_idx = 2
    self._fourth_sparse_stage_idx = 3

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
      self._sparsity_info[-1]._monitor_period = self._initial_monitor_period
    self._num_sparsity_info = len(self._sparsity_info)

  def update_stage(self, tensor_idx):
    if self._sparsity_info[tensor_idx]._sparsity < self._sparsity_threshold_list[self._first_sparse_stage_idx]:
      self._sparsity_info[tensor_idx]._stage = SparseStage.dense
    elif self._sparsity_info[tensor_idx]._sparsity >= self._sparsity_threshold_list[self._first_sparse_stage_idx] \
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

  def update_history(self, tensor_idx):
    # The tensor idx will always index a tensor enabling monitor
    current_history_length = len(self._sparsity_info[tensor_idx]._sparsity_history)
    # Get the right history length
    if self._status == Status.active:
      history_length = self._monitor_sparsity_history_length
    elif self._status == Status.hibernate:
      history_length = self._hibernation_sparsity_history_length
    if current_history_length == history_length:
      self._sparsity_info[tensor_idx]._sparsity_history.pop(0)
    self._sparsity_info[tensor_idx]._sparsity_history.append(self._sparsity_info[tensor_idx]._sparsity)

  def adjust_monitor_period(self, tensor_idx):
    # The tensor idx will always index a tensor enabling monitor
    current_history_length = len(self._sparsity_info[tensor_idx]._sparsity_history)
    history_length = self._monitor_sparsity_history_length
    if current_history_length == history_length:
      #enable the dynamic monitor period adjustment after N monitor period
      sparsity_diff = abs(self._sparsity_info[tensor_idx]._sparsity -\
                          self._sparsity_info[tensor_idx]._sparsity_history[0])
      # Only times the monitor period when condition is met and it is smaller
      # than the hibernation period
      if sparsity_diff < self._monitor_cond_sparsity_difference and \
          self._sparsity_info[tensor_idx]._monitor_period < self._hibernation_period:
        self._sparsity_info[tensor_idx]._monitor_period = \
            self._sparsity_info[tensor_idx]._monitor_period * self._monitor_period_multiple 
        if self._sparsity_info[tensor_idx]._monitor_period > self._hibernation_period:
          self._sparsity_info[tensor_idx]._monitor_period = self._hibernation_period
        
  def check_hibernation(self):
    assert self._can_hibernate == False
    hibernation_ready_valid_count = 0
    hibernation_ready_invalid_count = 0
    for i in range(self._num_sparsity_info):
      # Condition 1
      next_possible_monitor_period = self._sparsity_info[i]._monitor_period * self._monitor_period_multiple
      if next_possible_monitor_period > self._hibernation_period and self._sparsity_info[i]._valid:
        hibernation_ready_valid_count += 1

      # Condition 2
      #if self._sparsity_info[i]._sparsity < self._initial_sparsity and self._sparsity_info[i]._sparsity > 0:
      if not self._sparsity_info[i]._valid:
        hibernation_ready_invalid_count += 1

    total_count = hibernation_ready_valid_count + hibernation_ready_invalid_count
    if total_count == self._num_sparsity_info and hibernation_ready_invalid_count != self._num_sparsity_info:
      self._can_hibernate = True
      # Remove the sparsity history
      for i in range(self._num_sparsity_info):
        self._sparsity_info[i]._sparsity_history = []
    return total_count

  def check_active(self):
    assert self._can_hibernate == True
    history_length = self._hibernation_sparsity_history_length
    for i in range(self._num_sparsity_info):
      current_history_length = len(self._sparsity_info[i]._sparsity_history)
      if current_history_length == history_length:
        sparsity_diff = abs(self._sparsity_info[i]._sparsity -\
                            self._sparsity_info[i]._sparsity_history[0])
        # When detecting severe sparsity change
        if sparsity_diff > self._hibernation_cond_sparsity_difference:
          self._can_hibernate = False
          break

  def update_results(self, global_step, tensor_idx):
    stage = self._sparsity_info[tensor_idx]._stage
    if stage == SparseStage.dense:
      stage_str = "dense"
    elif stage == SparseStage.first_stage:
      stage_str = "first"
    elif stage == SparseStage.second_stage:
      stage_str = "second"
    elif stage == SparseStage.third_stage:
      stage_str = "third"
    elif stage == SparseStage.fourth_stage:
      stage_str = "fourth"
    else:
      return

    if self._status == Status.hibernate:
      status_str = "hibernation"
    elif self._status == Status.active:
      status_str = "active"

    name = self._sparsity_info[tensor_idx]._data_tensor.name
    sparsity = self._sparsity_info[tensor_idx]._sparsity
    ratio = self._sparsity_info[tensor_idx]._zero_block_ratio
    monitor_period = self._sparsity_info[tensor_idx]._monitor_period

    self._sparsity_info[tensor_idx]._results_str = (self._sparsity_info[tensor_idx]._results_str %\
                                                   (name, global_step, status_str, stage_str, monitor_period, sparsity, ratio))

  def animate(self, i, ax, data_dict):
    cmap = ListedColormap(['black', 'red'])
    label = 'Local step in monitoring period: {0}'.format(i)
    matrix = data_dict[i] 
    mesh = ax.pcolormesh(matrix, cmap=cmap)
    ax.set_xlabel(label)
    return mesh,

  def results_io(self, tensor_idx, workpath, enable_file_io, model):
    if model == '':
      print('The model parameter is empty')
      exit()
    stage = self._sparsity_info[tensor_idx]._stage
    
    if stage == SparseStage.first_stage:
      stage_str = "first"
    elif stage == SparseStage.second_stage:
      stage_str = "second"
    elif stage == SparseStage.third_stage:
      stage_str = "third"
    elif stage == SparseStage.fourth_stage:
      stage_str = "fourth"
    else:
      return

    if enable_file_io:
      # Output results
      enable_results = False
      if enable_results:
        tensor_name = self._sparsity_info[tensor_idx]._data_tensor.name
        file_name = tensor_name.replace('/', '_').replace(':', '_') + '.txt'
        self._sparsity_info[tensor_idx]._fd = open(workpath+'/'+file_name, 'a')
        self._sparsity_info[tensor_idx]._fd.write(self._sparsity_info[tensor_idx]._results_str)
        self._sparsity_info[tensor_idx]._fd.close()

      # Output data
      enable_data_file = True
      if enable_data_file:
        # Only deal with the first data within a monitor interval
        tensor_name = self._sparsity_info[tensor_idx]._data_tensor.name
        tensorkey = ''
        if model == 'alexnet' or model == 'vggnet':
          if tensorname_regex.match(tensor_name):
            tensorkey = tensorname_regex.match(tensor_name).group(1)
        elif model == 'resnet':
          tensorkey = tensor_name.split(":")[0]
        if tensorkey == '':
          print("Tensor name unavailable")
          exit()
        sparsities = problem_size_map[model][tensorkey]
        interval_threshold = 0.025
        for i in range(len(sparsities)):
          # Im2col increase sparsity a little bit
          sparsity_increment = 0.05
          if 'rb4' in tensorkey:
            sparsity_increment = 0.1
          elif 'rb1' in tensorkey:
            sparsity_increment = 0.01
          sp = sparsities[i]+sparsity_increment
          curr_sp = self._sparsity_info[tensor_idx]._sparsity
          curr_flag = self._sparsity_info[tensor_idx]._flag_map[model][tensorkey][i]
          if not curr_flag:
            if curr_sp < sp + interval_threshold and curr_sp >= sp - interval_threshold:
              data = self._sparsity_info[tensor_idx]._extracted_data_list[0]
              batch_size = data.shape[0]
              output_h = data.shape[1]
              output_w = data.shape[2]
              col_size = data.shape[3]
              data = np.reshape(data, batch_size*output_h*output_w*col_size)

              print("Print out: ", tensorkey)
              print("batch size: ", batch_size)
              print("output_h: ", output_h)
              print("output_w: ", output_w)
              print("col_size: ", col_size)
              file_name = model+'_'+tensorkey.replace('/', '_')+'_'+str(int(sparsities[i]*100))+'.data'
              #file_name = workpath+'/'+file_name
              if not os.path.isfile(file_name):
                self._sparsity_info[tensor_idx]._fd = open(file_name, 'w')
                data.tofile(self._sparsity_info[tensor_idx]._fd, "\n")
                self._sparsity_info[tensor_idx]._fd.close()
              self._sparsity_info[tensor_idx]._flag_map[model][tensorkey][i] = True

      # Enable gif generation
      enable_gif = False
      if enable_gif and not os.path.isfile(figure_name):
        # Plot the data pattern
        figure_name = tensor_name.replace('/', '_').replace(':', '_') +\
                      stage_str

        fig, ax = plt.subplots()
        num_frames = len(self._sparsity_info[tensor_idx]._extracted_data)
        ani = animation.FuncAnimation(fig, animate, frames=num_frames,
                                      fargs=(ax, self._sparsity_info[tensor_idx]._extracted_data,),
                                      interval=500, repeat=False, blit=True)                        
        
        ani.save(workpath+figure_name+'.gif', dpi=80, writer='imagemagick')
        plt.close('all')
    else:
      print (self._sparsity_info[tensor_idx]._results_str)

  def scheduler_before(self, global_step):
    # Manage the status
    if self._can_hibernate:
      self._status = Status.hibernate
    else:
      self._status = Status.active

    # Schedule the sparsity monitor
    if self._status == Status.active:
      for i in range(self._num_sparsity_info):
        if global_step % self._sparsity_info[i]._monitor_period == 0:
          self._sparsity_info[i]._enabled = True
          self._sparsity_info[i]._valid = True
    elif self._status == Status.hibernate:
      if global_step % self._hibernation_period == 0:
        for i in range(self._num_sparsity_info):
          self._sparsity_info[i]._enabled = True
          self._sparsity_info[i]._valid = True

    self._monitor_enabled_tensor_list = []
    self._monitor_enabled_tensor_id_list = []
    for i in range(self._num_sparsity_info):
      if self._sparsity_info[i]._enabled:
        self._monitor_enabled_tensor_list.append(self._sparsity_info[i]._data_tensor)
        self._monitor_enabled_tensor_list.append(self._sparsity_info[i]._sparsity_tensor)
        self._monitor_enabled_tensor_id_list.append(i)

    # Return the data list that needed for monitoring
    return self._monitor_enabled_tensor_list

  def scheduler_after(self, value_list, global_step, model, workpath="", enable_file_io=False):
    # Time CONSUMING part is here
    if self._mode == Mode.monitor:
      self.__collect_and_monitoring(value_list)

    for i in range(self._num_sparsity_info):
      if not self._sparsity_info[i]._enabled:
        continue

      # post processing of scheduler parameters
      # Turn off the monitoring
      if self._sparsity_info[i]._local_step == self._monitor_interval:
        self._sparsity_info[i]._enabled = False

        # Update the sparsity history based on configured history length
        self.update_history(i)

        # Update the sparsity stage of corresponding data
        self.update_stage(i)

        # Adjust the monitor period
        if self._status == Status.active:
          self.adjust_monitor_period(i)


        # Call file IO utility here
        self.update_results(global_step, i)
        self.results_io(i, workpath, enable_file_io, model)

        # Reset the sparsity info for next period
        self._sparsity_info[i].reset()
      elif self._sparsity_info[i]._local_step == 1 and self._sparsity_info[i]._sparsity < self._initial_sparsity:
        # The sparsity is too small to be recorded
        self._sparsity_info[i].reset()
        self._sparsity_info[i]._enabled = False
        self._sparsity_info[i]._valid = False

    # Status transition
    if self._status == Status.active:
      ready_count = self.check_hibernation()
    elif self._status == Status.hibernate:
      self.check_active()

  def __collect_and_monitoring(self, value_list):
    if len(value_list) == 0:
      return
    num_selected_tensor = len(self._monitor_enabled_tensor_id_list)
    assert num_selected_tensor == len(value_list)/2
    for tensor_id, i in zip(self._monitor_enabled_tensor_id_list, range(num_selected_tensor)):
      # indices
      data_idx = i * 2
      sparsity_idx = i * 2 + 1
      current_data = value_list[data_idx]
      current_sparsity = value_list[sparsity_idx]
      shape = self._sparsity_info[tensor_id]._data_tensor.get_shape()
      tensor = self._sparsity_info[tensor_id]._data_tensor
      batch_idx = 0
      channel_idx = -1
      stage = self._sparsity_info[tensor_id]._stage
      #if stage == SparseStage.dense:
      #  sparsity_threshold = 0.6
      #elif stage == SparseStage.first_stage:
      #  sparsity_threshold = 0.7
      #elif stage == SparseStage.second_stage:
      #  sparsity_threshold = 0.8
      #elif stage == SparseStage.third_stage or stage == SparseStage.fourth_stage:
      #  sparsity_threshold = 0.9
      sparsity_threshold = self._initial_sparsity

      #if (self._sparsity_info[tensor_id]._local_step == 0 and current_sparsity > sparsity_threshold)\
      #    or self._sparsity_info[tensor_id]._local_step > 0:
      if self._sparsity_info[tensor_id]._local_step < self._monitor_interval:
        if self._sparsity_info[tensor_id]._sparsity < 0:
          # The first iteration within the monitor interval
          self._sparsity_info[tensor_id]._sparsity = current_sparsity
        else:
          # Calculate the moving average
          self._sparsity_info[tensor_id]._sparsity = self._sparsity_info[tensor_id]._sparsity * 0.9 + current_sparsity * 0.1

        if current_sparsity > sparsity_threshold:
          block_size = self._sparsity_info[tensor_id]._zero_block_size
          zero_block_ratio = sparsity_util.zero_block_ratio_matrix(current_data, shape, block_size)
          if self._sparsity_info[tensor_id]._zero_block_ratio < 0:
            # The first iteration within the monitor interval
            self._sparsity_info[tensor_id]._zero_block_ratio = zero_block_ratio
          else:
            # Calculate the moving average
            self._sparsity_info[tensor_id]._zero_block_ratio = self._sparsity_info[tensor_id]._zero_block_ratio * 0.9 + zero_block_ratio * 0.1

          #self._sparsity_info[tensor_id]._extracted_data_list.append(
          #  sparsity_util.feature_map_extraction(current_data, self._data_format, batch_idx, channel_idx))
          
          self._sparsity_info[tensor_id]._extracted_data_list.append(current_data)
        self._sparsity_info[tensor_id]._local_step += 1
      else:
        continue
    
