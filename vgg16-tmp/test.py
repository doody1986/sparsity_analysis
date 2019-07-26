import tensorflow as tf
import inference as inf
import RecordInputImagePreprocessor.py
import datasets.py

 /spartan/imagenet
def preprocess(dataset):
    imageNetPath = '/spartan/imagenet/'
    imgNet = ImagenetData( )
    pp = RecordInputImagePreprocessor()
    pp.minibatch(imageNetPath, subset, use_data_set)

def randomTensor():
    a = tf.random.uniform(shape=[224,224,3,3])
    return a


inf.inference(randomTensor())
