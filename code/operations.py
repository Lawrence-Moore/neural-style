import tensorflow as tf
from vgg16 import vgg16

def content_loss(content_layer, generated_layer):
    # sess.run(vgg_net.image.assign(generated_image))

    # now we define the loss as the difference between the reference activations and 
    # the generated image activations in the specified layer
    # return 1/2 * tf.nn.l2_loss(content_layer - generated_layer)
    return tf.scalar_mul(.5, tf.nn.l2_loss(content_layer - generated_layer))

def style_loss(style_layers, generated_layers, weights):
    
    layer_losses = []
    for index in [0, 1, 2, 3]:
        reference_layer = style_layers[index]
        generated_image_layer = generated_layers[index]

        N = reference_layer.shape[3]
        M = reference_layer.shape[1] * reference_layer.shape[2]
        # layer_losses.append(weights[index] * (4 / (M**2 * N**2)) * tf.nn.l2_loss(get_gram_matrix(reference_layer, N) - get_gram_matrix(generated_image_layer, N)))
        layer_losses.append(tf.scalar_mul(weights[index] * 4 / (M**2 * N**2), tf.nn.l2_loss(get_gram_matrix(reference_layer, N) - get_gram_matrix(generated_image_layer, N))))

    return sum(layer_losses)


def get_gram_matrix(matrix, num_filters):
    # first vectorize the matrix
    matrix_vectorized = tf.reshape(matrix, [-1, num_filters])
    # then calculate the gram by multiplying the vector by its transpose
    return tf.matmul(tf.transpose(matrix_vectorized), matrix_vectorized)


# def run_vgg(sess, image):
#     print "making the template", image.shape
#     imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
#     net = vgg16(imgs, 'vgg16_weights.npz', sess)
#     print "model loaded"
#     # net = VGG16({'data': image})
#     # net.load(model_data_path, session)
#     # session.run(net.get_output(), feed_dict={input_node: image})
#     sess.run(net.probs, feed_dict={net.imgs: image})
#     return net


