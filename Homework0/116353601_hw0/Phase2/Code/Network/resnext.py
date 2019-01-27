
import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

#################################
previously crearted functions
#################################

def conv_layer_with_relu(net, num_filters = None, kernel_size = None, strides = None):
    net = tf.layers.conv2d(inputs=net, padding='same', filters=num_filters, kernel_size=kernel_size, activation=None, strides = strides)
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)

    return net

def conv_layer_without_relu(net, num_filters = None, kernel_size = None, strides = None):
    net = tf.layers.conv2d(inputs=net, padding='same', filters=num_filters, kernel_size=kernel_size, activation=None, strides = strides)
    net = tf.layers.batch_normalization(net)

    return net

def create_single_resnet_block(net, num_filters = None, kernel_size = None, prev_filter = None, downsample = False):
    if(prev_filter == num_filters):
        original_block = net
    else:
        original_block = conv_layer_without_relu(net, num_filters=num_filters, kernel_size=kernel_size, strides=1)

    if downsample:
        original_block = conv_layer_without_relu(net, num_filters=num_filters, kernel_size=kernel_size, strides=2)
        net = conv_layer_with_relu(net, num_filters=num_filters, kernel_size=kernel_size, strides=2)
    else:
        # original_block = conv_layer_with_relu(original_block, num_filters=num_filters, kernel_size=kernel_size, strides=1)
        net = conv_layer_with_relu(net, num_filters=num_filters, kernel_size=kernel_size, strides=1)

    net = conv_layer_without_relu(net, num_filters=num_filters, kernel_size=kernel_size, strides=1)

    net = tf.math.add(net, original_block)

    net = tf.nn.relu(net)

    return net

def n_resnet_blocks(num_blocks, net, prev_filter = None, num_filters = None, kernel_size = None, downsample = False):

    if downsample:
        net = create_single_resnet_block(net, num_filters = num_filters, kernel_size = kernel_size, prev_filter = prev_filter, downsample = True)
        for i in range(num_blocks-1):
            net = create_single_resnet_block(net, num_filters = num_filters, prev_filter = prev_filter, kernel_size = kernel_size)
    else:
        for i in range(num_blocks):
            net = create_single_resnet_block(net, num_filters = num_filters, prev_filter = prev_filter, kernel_size = kernel_size)
    return net

def MY_RESNET(Img, ImageSize, MiniBatchSize):
    num_classes = 10;

    net = Img
    net = conv_layer_with_relu(net=net, num_filters = 64, kernel_size = 3, strides = 1)
    net = n_resnet_blocks(2, net=net, prev_filter = 64, num_filters = 64, kernel_size = 3)
    net = n_resnet_blocks(2, net=net, prev_filter = 64, num_filters = 128, kernel_size = 3, downsample = True)
    net = n_resnet_blocks(3, net=net, prev_filter = 128, num_filters = 256, kernel_size = 3, downsample = True)

    net1 = tf.contrib.layers.flatten(net)

    net2 = tf.layers.dense(inputs=net1, name='layer_fc1',
                      units=128, activation=tf.nn.relu)
    #Contruct 2nd Fully-connected layer
    net3 = tf.layers.dense(inputs=net2, name='layer_fc_out',
                      units=num_classes, activation=None)

    prLogits = net3
    prSoftMax = tf.nn.softmax(net3)
    return prLogits, prSoftMax



#################################

def MY_ResneXt():
	
	
	

	net = tf.contrib.layers.flatten(net)

    net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=128, activation=tf.nn.relu)
    #Contruct 2nd Fully-connected layer
    net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)

	prLogits = net
	prSoftMax = tf.nn.softmax(net)
	return prLogits, prSoftMax