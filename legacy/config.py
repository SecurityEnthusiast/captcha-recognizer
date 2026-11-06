"""Constants for the TensorFlow 1 scripts in this folder.

Run those scripts from this directory so ``import config`` resolves here.
They are not the supported training path.
"""

IMAGE_HEIGHT = 48
IMAGE_WIDTH = 128
CHAR_SET = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CLASSES_NUM = len(CHAR_SET)
CHARS_NUM = 5
RECORD_DIR = "./data"
TRAIN_FILE = "train.tfrecords"
VALID_FILE = "valid.tfrecords"
