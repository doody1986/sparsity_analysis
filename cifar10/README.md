# What we implemented?

I have consulted the following implementations to finally bring resnet to life:

```
https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/resnet.py
https://github.com/chenxi116/TF-resnet/blob/master/resnet_model.py
https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
```

The model implemented here is more similar to:

```
https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/resnet.py
```

# How to run?

In order to run:

```
# for 100k iterations this will take ~13h in an core i7-8700 
python resnet50_train.py
```

# What results are observed?

There are some logs and models available.
```
# 14 layer resnet, with regular channel sizes 64 128 256, trained ~11k iter
tensorboard --logdir=tensorboard-logs/resnet-1-4-4-4-1-regular-channels/

# 8 layer resnet, with  reduced channel sizes 32 64 128, trained 100k iter
tensorboard --logdir=tensorboard-logs/resnet-1-2-2-2-1-low-channels/

# Shi's net, trained 10k iter
tensorboard --logdir=tensorboard-logs/cifar10_train-10k-detailed/
```

# Important files?
To modify the resnet to bigger sizes, modify this file `resnet50.py` and this
function:

```
# 'n' changes the number of consecutive blocks
def inference(input_tensor_batch, n=2, reuse=False):
```

Right now, the network has this many total layers:
```
total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
```

The last block with channels of 512 in size is omitted, but can be integrated.
