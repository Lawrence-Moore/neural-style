import numpy as np
import tensorflow as tf

'''
Based off the cafe to tensorflow code and https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
'''


class Network(object):

    def __init__(self, input, weight_path):
        # The input nodes for this network
        self.weights = np.load(weight_path)
        self.result_dict = {}
        self.update("initial image", input)
        self.setup()

    def update(self, name, result):
        self.previous_result = result
        self.result_dict[name] = result

    def conv(self, name):
        with tf.variable_scope(name):

            # get the weights and biases
            weights = self.get_weights(name)
            biases = self.get_biases(name)

            # convolve
            conv = tf.nn.conv2d(self.previous_result, weights, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)

            # relu
            relu = tf.nn.relu(bias)
            self.update(name, relu)
            return self

    def max_pool(self, name):
        pool = tf.nn.max_pool(self.previous_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        self.update(name, pool)
        return self

    def get_weights(self, name):
        return self.weights[name + '_W'].astype('float32')

    def get_biases(self, name):
        return self.weights[name + '_b'].astype('float32')


