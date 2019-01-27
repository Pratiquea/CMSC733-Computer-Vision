"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Prateek Arora(pratique@terpmail.umd.edu)
Robotics Graduate Student
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True


def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    
    # ###########################
    # ###### flow of code #######
    # ###########################
    # #   -- Construct 1st convolution layer
    # #   -- Construct 2nd convolution layer
    # #   -- flaten the ouput of 2nd layer to proper input format of 1st Fully-connected layer 
    # #   -- Contruct 1st Fully-connected layer
    # #   -- Contruct 2nd Fully-connected layer


    #############################
    # Fill your network here!
    #############################
    
    #Convolution layer 1
    filter_size1 = 3      #filter size for first convolution layer(i.e. 5x5 pixels)
    num_filters1 = 8     #number of filters in first convolution layer

    #Convolution layer 2
    filter_size2 = 3      #filter size for second convolution layer(i.e. 5x5 pixels)
    num_filters2 = 128     #number of filters in second convolution layer

    # Fully-connected layer.
    fc1_size = 256             # Number of neurons in fully-connected layer.
    fc2_size = 128             # Number of neurons in fully-connected layer.
    num_classes = 10

        
    net = Img
    #Constructing first convolution layer
    net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=num_filters1, kernel_size=filter_size1, activation=tf.nn.relu)
    layer_conv1 = net
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    #Constructing second convolution layer
    net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=num_filters2, kernel_size=filter_size2, activation=tf.nn.relu)
    
    layer_conv2 = net
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.contrib.layers.flatten(net)
    #Contruct 1st Fully-connected layer
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=fc1_size, activation=tf.nn.relu)
    #Contruct 2nd Fully-connected layer
    net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)

    prLogits = net
    
    prSoftMax = tf.nn.softmax(net)


    return prLogits, prSoftMax


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



def SIMPLE_NET(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    
    # ###########################
    # ###### flow of code #######
    # ###########################
    # #   -- Construct 1st convolution layer
    # #   -- Construct 2nd convolution layer
    # #   -- flaten the ouput of 2nd layer to proper input format of 1st Fully-connected layer 
    # #   -- Contruct 1st Fully-connected layer
    # #   -- Contruct 2nd Fully-connected layer


    #############################
    # Fill your network here!
    #############################
    
    #Convolution layer 1
    filter_size1 = 5      #filter size for first convolution layer(i.e. 5x5 pixels)
    num_filters1 = 32     #number of filters in first convolution layer

    #Convolution layer 2
    filter_size2 = 5      #filter size for second convolution layer(i.e. 5x5 pixels)
    num_filters2 = 64     #number of filters in second convolution layer

    #Convolution layer 3
    filter_size3 = 5      
    num_filters3 = 64     

    #Convolution layer 4
    filter_size4 = 5     
    num_filters4 = 128   


    # Fully-connected layer.
    fc1_size = 256             # Number of neurons in fully-connected layer.
    fc2_size = 128             # Number of neurons in fully-connected layer.
    num_classes = 10

        
    net = Img
    
    ### 1
    net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=num_filters1, kernel_size=filter_size1, activation=None)
    net = tf.layers.batch_normalization(net, name='bn1T')
    net = tf.nn.relu(net, name='relu1T')
    
    # net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    

    ### 2
    net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=num_filters2, kernel_size=filter_size2, activation=None)
    net = tf.layers.batch_normalization(net, name='bn2T')
    net = tf.nn.relu(net, name='relu2T')
    
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    

    ### 3
    net = tf.layers.conv2d(inputs=net, name='layer_conv3', padding='same',
                       filters=num_filters3, kernel_size=filter_size3, activation=None)
    net = tf.layers.batch_normalization(net, name='bn3T')
    net = tf.nn.relu(net, name='relu3T')
    
    # net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


    ### 4
    net = tf.layers.conv2d(inputs=net, name='layer_conv4', padding='same',
                       filters=num_filters4, kernel_size=filter_size4, activation=None)
    net = tf.layers.batch_normalization(net, name='bn4T')
    net = tf.nn.relu(net, name='relu4T')
    
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


    #flattening layer
    net = tf.contrib.layers.flatten(net)
    


    #Contruct 1st Fully-connected layer
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=fc1_size, activation=tf.nn.relu)
    #2
    net = tf.layers.dense(inputs=net, name='layer_fc2',
                      units=fc2_size, activation=tf.nn.relu)
    #3
    net = tf.layers.dense(inputs=net, name='layer_fc3',
                      units=64, activation=tf.nn.relu)
    #4
    net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)

    prLogits = net
    
    prSoftMax = tf.nn.softmax(net)


    return prLogits, prSoftMax

