import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
NUM_CLASSES = 20

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
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


def inference(images):
    """Build the VGG-16 model.
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
        monitored_tensor_list.append(conv1_1)

    # conv 1.2
    with tf.variable_scope('conv1_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv1_1, biases)
        conv1_2 = tf.nn.relu(pre_activation, name="relu")
        monitored_tensor_list.append(conv1_2)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    monitored_tensor_list.append(pool1)
    
    # conv 2.1
    with tf.variable_scope('conv2_1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(pre_activation, name="relu")
        monitored_tensor_list.append(conv2_1)

    # conv 2.2
    with tf.variable_scope('conv2_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(pre_activation, name="relu")
        monitored_tensor_list.append(conv2_2)

    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    monitored_tensor_list.append(pool2)


    # conv 3.1
    with tf.variable_scope('conv3_1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
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
        monitored_tensor_list.append(conv3_2)

    # conv 3.3
    with tf.variable_scope('conv3_3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, 256], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(pre_activation, name="relu")
        monitored_tensor_list.append(conv3_3)

    # pool3
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    monitored_tensor_list.append(pool3)

    # conv 4.1
    with tf.variable_scope('conv4_1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 512], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
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
        monitored_tensor_list.append(conv4_2)

    # conv 4.3
    with tf.variable_scope('conv4_3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 512, 512], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(pre_activation, name="relu")
        monitored_tensor_list.append(conv4_3)

    # pool4
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    monitored_tensor_list.append(pool4)

    # conv 5.1
    with tf.variable_scope('conv5_1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(pre_activation, name="relu")
        monitored_tensor_list.append(conv5_1)

    # conv 5.2
    with tf.variable_scope('conv5_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(pre_activation, name="relu")
        monitored_tensor_list.append(conv5_2)

    # conv 5.3
    with tf.variable_scope('conv5_3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 512, 512], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(pre_activation, name="relu")
        monitored_tensor_list.append(conv5_3)

    # pool5
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    monitored_tensor_list.append(pool5)

    # dense1
    with tf.variable_scope('dense1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool5, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
        dense1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="relu")
        monitored_tensor_list.append(dense1)
        

    # dense2
    with tf.variable_scope('dense2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4096, 1000], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0.1))
        dense2 = tf.nn.relu(tf.matmul(dense1, weights) + biases, name="relu")
        monitored_tensor_list.append(dense2)

    # dense3
    with tf.variable_scope('dense3') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1000, 1000], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0.1))
        dense3 = tf.nn.relu(tf.matmul(dense2, weights) + biases, name="relu")
        monitored_tensor_list.append(dense3)

    # softmax layer
    with tf.variable_scope('softmax') as scope:
        weights = _variable_with_weight_decay('weights', [1000, NUM_CLASSES], stddev=1/1000.0, wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax = tf.add(tf.matmul(dense3, weights), biases, name="output")

    return softmax, monitored_tensor_list


