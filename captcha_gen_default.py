from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import os
from captcha.image import ImageCaptcha

import config

IMAGE_HEIGHT = config.IMAGE_HEIGHT
IMAGE_WIDTH = config.IMAGE_WIDTH
CHARS_NUM = config.CHARS_NUM
CHAR_SET = config.CHAR_SET

TEST_SIZE = 1000
TRAIN_SIZE = 50000
VALID_SIZE = 20000

FLAGS = None

def gen(gen_dir, total_size, chars_num):
  if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)
  image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT,font_sizes=[40])
  for i in range(total_size):
    label = ''.join(random.sample(CHAR_SET, CHARS_NUM))
    image.write(label, os.path.join(gen_dir, label+'_num'+str(i)+'.png'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test_dir',
      type=str,
      default='./data/test_data',
      help='Test data-set directory address'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./data/train_data',
      help='Train data-set directory adddress'
  )
  parser.add_argument(
      '--valid_dir',
      type=str,
      default='./data/valid_data',
      help='Validation data-set directory address'
  )
  FLAGS, unparsed = parser.parse_known_args()
  print('>> generate %d captchas in %s' % (TEST_SIZE, FLAGS.test_dir))
  gen(FLAGS.test_dir, TEST_SIZE, CHARS_NUM)
  print ('>> generate %d captchas in %s' % (TRAIN_SIZE, FLAGS.train_dir))
  gen(FLAGS.train_dir, TRAIN_SIZE, CHARS_NUM)
  print ('>> generate %d captchas in %s' % (VALID_SIZE, FLAGS.valid_dir))
  gen(FLAGS.valid_dir, VALID_SIZE, CHARS_NUM)
  print ('>> generate Done!')