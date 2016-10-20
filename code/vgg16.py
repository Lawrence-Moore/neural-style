########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from network import Network

# based on https://www.cs.toronto.edu/~frossard/post/vgg16/

class vgg16(Network):
    def setup(self):
        (self.conv(name='conv1_1')
             .conv(name='conv1_2')
             .max_pool(name='pool1')
             .conv(name='conv2_1')
             .conv(name='conv2_2')
             .max_pool(name='pool2')
             .conv(name='conv3_1')
             .conv(name='conv3_2')
             .conv(name='conv3_3')
             .max_pool(name='pool3')
             .conv(name='conv4_1')
             .conv(name='conv4_2')
             .conv(name='conv4_3')
             .max_pool(name='pool4')
             .conv(name='conv5_1')
             .conv(name='conv5_2')
             .conv(name='conv5_3')
             .max_pool(name='pool5'))

    def get_layers(self):
        return [self.result_dict['conv1_1'], self.result_dict['conv2_1'], self.result_dict['conv3_1'], self.result_dict['conv4_1'], self.result_dict['conv5_1']]
