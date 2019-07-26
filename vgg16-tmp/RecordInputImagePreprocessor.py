class RecordInputImagePreprocessor(object):
  """Preprocessor for images with RecordInput format."""

  def __init__(self,
               height,
               width,
               batch_size,
               device_count,
               dtype,
               train,
               distortions,
               resize_method,
               shift_ratio):
    self.height = height
    self.width = width
    self.batch_size = batch_size
    self.device_count = device_count
    self.dtype = dtype
    self.train = train
    self.resize_method = resize_method
    self.shift_ratio = shift_ratio
    self.distortions = distortions
    if self.batch_size % self.device_count != 0:
      raise ValueError(
          ('batch_size must be a multiple of device_count: '
           'batch_size %d, device_count: %d') %
          (self.batch_size, self.device_count))
    self.batch_size_per_device = self.batch_size // self.device_count

  def preprocess(self, image_buffer, bbox, thread_id):
    """Preprocessing image_buffer using thread_id."""
    image = tf.image.decode_jpeg(image_buffer, channels=3,
                                 dct_method='INTEGER_FAST')
    if self.train:
      image = train_image(image, self.height, self.width, bbox, thread_id,
                          self.resize_method, self.distortions)
    else:
      image = eval_image(image, self.height, self.width, bbox, thread_id,
                         self.resize_method)
    # Note: image is now float32 [height,width,3] with range [0, 255]

    # image = tf.cast(image, tf.uint8) # HACK TESTING

    return image

  def parse_and_preprocess(self, value, counter):
    image_buffer, label_index, bbox, _ = parse_example_proto(value)
    image = self.preprocess(image_buffer, bbox, counter % 4)
    return (label_index, image)

  def minibatch(self, dataset, subset, use_data_sets):
    with tf.name_scope('batch_processing'):
      images = [[] for i in range(self.device_count)]
      labels = [[] for i in range(self.device_count)]
      if use_data_sets:
        file_names = glob.glob(dataset.tf_record_pattern(subset))
        batch_size_per = self.batch_size / self.device_count
        num_threads = 10
        output_buffer_size = num_threads * 2000

        counter = tf.contrib.data.Dataset.range(sys.maxint)
        ds = tf.contrib.data.TFRecordDataset(file_names)
        ds = tf.contrib.data.Dataset.zip((ds, counter))
        ds = ds.map(
            self.parse_and_preprocess,
            num_threads=num_threads,
            output_buffer_size=output_buffer_size)
        shuffle_buffer_size = 10000
        ds = ds.shuffle(shuffle_buffer_size)
        repeat_count = -1  # infinite repetition
        ds = ds.repeat(repeat_count)
        ds = ds.batch(batch_size_per)
        ds_iterator = ds.make_one_shot_iterator()

        for d in xrange(self.device_count):
          labels[d], images[d] = ds_iterator.get_next()

      else:
        # Build final results per device.
        record_input = data_flow_ops.RecordInput(
            file_pattern=dataset.tf_record_pattern(subset),
            seed=301,
            parallelism=64,
            buffer_size=10000,
            batch_size=self.batch_size,
            shift_ratio=self.shift_ratio,
            name='record_input')
        records = record_input.get_yield_op()
        records = tf.split(records, self.batch_size, 0)
        records = [tf.reshape(record, []) for record in records]
        for i in xrange(self.batch_size):
          value = records[i]
          (label_index, image) = self.parse_and_preprocess(value, i % 4)
          device_index = i % self.device_count
          images[device_index].append(image)
          labels[device_index].append(label_index)

      label_index_batch = [None] * self.device_count
      for device_index in xrange(self.device_count):
        if use_data_sets:
          label_index_batch[device_index] = labels[device_index]
        else:
          images[device_index] = tf.parallel_stack(images[device_index])
          label_index_batch[device_index] = tf.concat(labels[device_index], 0)
        images[device_index] = tf.cast(images[device_index], self.dtype)
        depth = 3
        images[device_index] = tf.reshape(
            images[device_index],
            shape=[self.batch_size_per_device, self.height, self.width, depth])
        label_index_batch[device_index] = tf.reshape(
            label_index_batch[device_index], [self.batch_size_per_device])
        if FLAGS.summary_verbosity >= 2:
          # Display the training images in the visualizer.
          tf.summary.image('images', images)

      return images, label_index_batch


