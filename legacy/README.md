# Legacy scripts

These files are the original TensorFlow 1 training path. They use queue runners (`string_input_producer`, `TFRecordReader`, `Session`, `Saver`) that current TensorFlow removed. They are not part of the quick start, and this milestone does not make them run.

`import config` no longer resolves from these files. Do not point a new script at them. In particular, do not bring back the `rmtree` of the training directory from `captcha_train.py` and `captcha_multi_gpu_train.py`.

Use `captcha-demo` instead.
