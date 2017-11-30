# coding: utf-8
"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

import tensorflow as tf

flags = tf.app.flags

############################
#    environment setting   #
############################
flags.DEFINE_string('data_dir', '../data', 'path for mnist dataset')
flags.DEFINE_string('result_dir', '../results', 'path for saving results')
flags.DEFINE_string('model_dir', '../models', 'path for saving models')


############################
#    hyper parameters      #
############################
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('image_width', 32, 'image width')
flags.DEFINE_integer('image_height', 32, 'image height')
flags.DEFINE_integer('crop_width', 24, 'image width')
flags.DEFINE_integer('crop_height', 24, 'image height')
flags.DEFINE_integer('num_channels', 3, 'image channel')

# Exponential Learning Rate Decay Params
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_float('lr_decay', 0.9, 'learning rate')
                   
flags.DEFINE_integer('num_targets', 10, 'target size')
flags.DEFINE_integer('epochs', 50000, 'epoch')
flags.DEFINE_integer('train_sum_freq', 50, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('test_sum_freq', 500, 'the frequency of saving test summary(step)')
flags.DEFINE_integer('num_gens_to_wait', 250, 'num_gens_to_wait')

flags.DEFINE_integer('save_freq', 1000, 'the frequency of saving model(epoch)')
flags.DEFINE_boolean('if_showplt', False, 'show plt picture')


############################
#    mnist net setting     #
############################



cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
