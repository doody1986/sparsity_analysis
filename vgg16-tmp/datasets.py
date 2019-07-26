def create_dataset(data_dir, data_name):
  """Create a Dataset instance based on data_dir and data_name."""
  supported_datasets = {
      'synthetic': SyntheticData,
      'imagenet': ImagenetData,
      'cifar10': Cifar10Data,
  }
  if not data_dir:
    data_name = 'synthetic'

  if data_name is None:
    for supported_name in supported_datasets:
      if supported_name in data_dir:
        data_name = supported_name
        break

  if data_name is None:
    raise ValueError('Could not identify name of dataset. '
                     'Please specify with --data_name option.')

  if data_name not in supported_datasets:
    raise ValueError('Unknown dataset. Must be one of %s', ', '.join(
        [key for key in sorted(supported_datasets.keys())]))

  return supported_datasets[data_name](data_dir)


class Dataset(object):
  """Abstract class for cnn benchmarks dataset."""

  def __init__(self, name, height=None, width=None, depth=None, data_dir=None,
               queue_runner_required=False):
    self.name = name
    self.height = height
    self.width = width
    self.depth = depth or 3
    self.data_dir = data_dir
    self._queue_runner_required = queue_runner_required

  def tf_record_pattern(self, subset):
    return os.path.join(self.data_dir, '%s-*-of-*' % subset)

  def reader(self):
    return tf.TFRecordReader()

  @abstractmethod
  def num_classes(self):
    pass

  @abstractmethod
  def num_examples_per_epoch(self, subset):
    pass

  def __str__(self):
    return self.name

  def get_image_preprocessor(self):
    return None

  def queue_runner_required(self):
    return self._queue_runner_required


class ImagenetData(Dataset):
  """Configuration for Imagenet dataset."""

  def __init__(self, data_dir=None):
    if data_dir is None:
      raise ValueError('Data directory not specified')
    super(ImagenetData, self).__init__('imagenet', 300, 300, data_dir=data_dir)

  def num_classes(self):
    return 1000

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return 1281167
    elif subset == 'validation':
      return 50000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def get_image_preprocessor(self):
    return preprocessing.RecordInputImagePreprocessor

