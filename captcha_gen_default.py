from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import os
from captcha.image import ImageCaptcha

import config

TEST_SIZE = 1000
TRAIN_SIZE = 50000
VALID_SIZE = 20000

#FLAGS = None

def gen(gen_dir, total_size, chars_num):
  if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)
  image = ImageCaptcha(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,font_sizes=[40])
  for i in range(total_size):
    label = ''.join(random.sample(config.CHAR_SET, config.CHARS_NUM))
    image.write(label, os.path.join(gen_dir, label+'_num'+str(i)+'.png'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Change default value')

  parser.add_argument('--tedr',type=str,default='./data/test_data',help='Test data-set directory address')
  
  parser.add_argument('--tesz',type=int,default=TEST_SIZE,help='Number of test data-set')
  
  parser.add_argument('--trdr',type=str,default='./data/train_data',help='Train data-set directory adddress')
  
  parser.add_argument('--trsz',type=int,default=TRAIN_SIZE,help='Number of train data-set')
  
  parser.add_argument('--vadr',type=str,default='./data/valid_data',help='Validation data-set directory address')
  
  parser.add_argument('--vasz',type=int,default=VALID_SIZE,help='Number of validation data-set')


  FLAGS, unparsed = parser.parse_known_args()
  print('>> generate %d captchas in %s' % (FLAGS.tesz, FLAGS.tedr))
  gen(FLAGS.tedr, FLAGS.tesz, config.CHARS_NUM)
  print ('>> generate %d captchas in %s' % (FLAGS.trsz, FLAGS.trdr))
  gen(FLAGS.trdr, FLAGS.trsz, config.CHARS_NUM)
  print ('>> generate %d captchas in %s' % (FLAGS.vasz, FLAGS.vadr))
  gen(FLAGS.vadr, FLAGS.vasz, config.CHARS_NUM)
  print ('>> generate Done!')