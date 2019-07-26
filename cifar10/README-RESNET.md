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

In order to run:

```
# for 100k iterations this will take ~13h in an core i7-8700 
python resnet50_train.py
```
