import tensorflow as tf
import numpy as np
import scipy
import argparse
from helper import read_image, save_image, generate_white_noise_image
from operations import content_loss, style_loss
from vgg16 import vgg16


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('alpha', 0.001, 'weight of the content loss')
flags.DEFINE_float('beta', 1, 'weight of the style loss')
flags.DEFINE_integer('iterations', 1000, 'number of iterations')



def stylize_image(content_image, style_image):
    layer_weights = [0.5, 1, 1.5, 3.0, 4.0]
    weights_path = 'vgg16_weights.npz'

    with tf.Session() as sess:

        # first load the model
        # generated_image = tf.Variable(np.random.uniform(-50, 50, (1, 224, 224, 3)).astype('float32'), name='generated_image', trainable=True)
        generated_image = tf.Variable(tf.constant(np.array(content_image)), name='generated_image')

        # then generate static white image and use the loss function to get a pretty picture
        
        # generate the content and style losses
        print "Running the network on the content image"
        vgg_net = vgg16(tf.constant(content_image), weights_path)
        content_layers = sess.run(vgg_net.get_layers())


        print "Running the network on the style image"
        vgg_net = vgg16(tf.constant(style_image), weights_path)
        style_layers = sess.run(vgg_net.get_layers())


        print "Time to style the image"
        sess.run(tf.initialize_all_variables())
        vgg_net = vgg16(generated_image, weights_path)
        generated_layers = vgg_net.get_layers()


        print "calculate loss functions"
        c_loss = content_loss(content_layers[4], generated_layers[4])
        s_loss = style_loss(style_layers, generated_layers, layer_weights)

        overall_loss = FLAGS.alpha * c_loss + FLAGS.beta * s_loss

        # use adam optimizer to minimize cost function
        print "optimize"
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(overall_loss)

        # make summaries
        sess.run(tf.initialize_all_variables())
        tf.scalar_summary("content_loss", c_loss)
        tf.scalar_summary("style_loss", s_loss)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('../logs', sess.graph)


        for step in range(FLAGS.iterations):
            sess.run(train_step)
            print "iteration: ", step
            if (step) % 10 == 0 or (step + 1) == FLAGS.iterations:
                print "content cost", sess.run(c_loss)
                print "style cost", sess.run(s_loss)
                print "Overall cost", sess.run(overall_loss)
                filename = 'output/%d.png' % (step)
                save_image(filename, sess.run(generated_image))


def main(argv):
    parser = argparse.ArgumentParser(description='Takes in the style and content image along with the location of the vgg model')
    parser.add_argument('--style', help='sum the integers (default: find the max)')
    parser.add_argument('--content', help='sum the integers (default: find the max)')
    args = parser.parse_args()
    if args.style and args.content:
        print "loading up images"
        content_image = read_image(args.content)
        style_image = read_image(args.style)

        print "get ready for style time"
        stylize_image(content_image, style_image)

if __name__=='__main__':
    tf.app.run()

