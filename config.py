# Captcha image properties
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 128
CHAR_SET = 'abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'
CLASSES_NUM = len(CHAR_SET)
CHARS_NUM = 5

# Train data folder and records addresses
RECORD_DIR = './data'
TRAIN_FILE = 'train.tfrecords'
VALID_FILE = 'valid.tfrecords'

    