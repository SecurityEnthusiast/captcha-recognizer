Introduce
=========
Image recognition captchas using TensorFlow, no need image segmentation, run on Ubuntu 18.04, Python 3.10, Tensorflow 2.10.0, Numpy 1.23.4

![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/1ab2s_num286.jpg)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/1ezx8_num398.jpg)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/1iv22_num346.jpg)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/1kxw2_num940.jpg)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/3mtj9_num765.jpg)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/1vuy5_num17.jpg)

Accuracy is directly related to generated Training data size and the Learning steps number.
For example, 
captcha generator: https://github.com/Gregwar/CaptchaBundle
 
Dependence
==========
### python 2.7
### Anaconda2 4.3.1
https://www.continuum.io/downloads#linux
### TensorFlow 1.1
https://github.com/tensorflow/tensorflow
### captcha
https://pypi.python.org/pypi/captcha/0.1.1

Usage
=====
## 1.prepare captchas
put your own captchas in **<current_dir>/data/train_data/** for training, **<current_dir>/data/valid_data/** for evaluating and **<current_dir>/data/test_data/** for recognize testing, images file name must be **label_\*.jpg** or **label_\*.png** and recommend size **128x48**. you can also use default generation:
```
python captcha_gen_default.py
```

## 2.convert dataset to tfrecords
the result file will be **<current_dir>/data/train.tfrecord** and **<current_dir>/data/valid.tfrecord**
```
python captcha_records.py
```

## 3.training
train and evaluate neural network on CPU or one single GPU
```
python captcha_train.py
```
you can also train over multiple GPUs
```
python captcha_multi_gpu_train.py
```

## 4.evaluate
```
python captcha_eval.py
```

## 5.recognize
read captchas from **<current_dir>/data/test_data/** for recogition
```
python captcha_recognize.py
```
result like this
```
...
image WFPMX_num552.png recognize ----> 'WFPMX'
image QUDKM_num468.png recognize ----> 'QUDKM'
```

