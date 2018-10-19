from convnets.module import conv2d
import tensorflow as tf

def vgg_encode(image,train_phas=True,img_channel=3):
    with tf.variable_scope('vgg'):
        net=conv2d('c1',image,img_channel,64,train_phas)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2])
        net = conv2d('c2',net, 64,128, train_phas)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2])
        net = conv2d('c3',net, 128,256, train_phas)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2])
        net = conv2d('c4', net, 256,256, train_phas)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2])
        net = conv2d('c5', net, 256, 512, train_phas)
        net = tf.layers.max_pooling2d(net, [2, 2], [1, 2])
        net = conv2d('c6', net, 512,512, train_phas)
        net = conv2d('c7', net, 512, 512, train_phas)
        print(net)
    return net