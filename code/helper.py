import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave

# values to center for vgg
MEAN_VALUES = np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))


def read_image(file_path):
    '''
    Takes in file path pointing to the image.
    Returns a tensorflow placeholder
    '''
    image = imread(file_path)
    image = imresize(image, (224, 224))

    # reside to 224x224x3
    image = image - MEAN_VALUES
    # image = image[None, ...]

    # tack on the extra dimension
    # images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # Input to the VGG model expects the mean to be subtracted.
    return image.astype(np.float32)

def save_image(path, image):
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    imsave(path, image)

def generate_white_noise_image():
    return tf.Variable(np.random.uniform(-50, 50, (1, 224, 224, 3)).astype('float32'), name='generated_image', trainable=True)
    # return np.random.uniform(-50, 50, (1, 224, 224, 3)).astype('float32')
