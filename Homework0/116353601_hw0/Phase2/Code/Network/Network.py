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
from tflearn.layers.conv import global_avg_pool
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
        original_block = tf.layers.max_pooling2d(inputs=original_block, pool_size=2, strides=2)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        # original_block = conv_layer_without_relu(original_block, num_filters=num_filters, kernel_size=kernel_size, strides=2)
        # net = conv_layer_with_relu(net, num_filters=num_filters, kernel_size=kernel_size, strides=2)
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
    net = conv_layer_with_relu(net=net, num_filters = 32, kernel_size = 3, strides = 1)
    net = n_resnet_blocks(5, net=net, prev_filter = 32, num_filters = 32, kernel_size = 3)
    net = n_resnet_blocks(3, net=net, prev_filter = 32, num_filters = 64, kernel_size = 3, downsample = True)
    net = n_resnet_blocks(2, net=net, prev_filter = 64, num_filters = 128, kernel_size = 3, downsample = True)

    net1 = tf.contrib.layers.flatten(net)

    net2 = tf.layers.dense(inputs=net1, name='layer_fc1',
                      units=128, activation=tf.nn.relu)
    #Contruct 2nd Fully-connected layer
    net3 = tf.layers.dense(inputs=net2, name='layer_fc_out',
                      units=num_classes, activation=None)

    prLogits = net3
    prSoftMax = tf.nn.softmax(net3)
    return prLogits, prSoftMax




# def transform_layer(net, strides, scope, depth):
#     with tf.name_scope(scope) :
#         net = conv_layer_with_relu(net=net, num_filters = depth, kernel_size = [1,1], strides = strides)

#         net = conv_layer_with_relu(net=net, num_filters = depth, kernel_size = [3,3], strides = strides)
#         return net





# def split_layer_and_create_res_blocks(net, strides, layer_name, depth, cardinality):
#     with tf.name_scope(layer_name) :
#         layers_split = list()
#         for i in range(cardinality) :
#             splits = transform_layer(net=net, strides=strides, scope=layer_name + '_splitN_' + str(i), depth=depth)
#             layers_split.append(splits)

#         return tf.concat(layers_split, axis=3)


# def residual_layer(net, out_dim, layer_num, res_block, depth, cardinality):
#     for i in range(res_block):
#         # input_dim = net.get_shape().as_list()[-1]
#         input_dim = int(np.shape(net)[-1])

#         if input_dim * 2 == out_dim:
#             flag = True
#             strides = 2
#             channel = input_dim // 2
#         else:
#             flag = False
#             strides = 1
#         x = split_layer_and_create_res_blocks(net, strides=strides, layer_name='split_layer_'+layer_num+'_'+str(i), depth=depth, cardinality=cardinality)
#         x = conv_layer_without_relu(net, num_filters = out_dim, kernel_size = 1, strides = 1)

#         if flag is True :
#             pad_input_x = tf.layers.average_pooling2d(inputs=net, pool_size=[2,2], strides=2, padding='SAME')            
#             pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
#         else :
#             pad_input_x = net

#         net = tf.nn.relu(x + pad_input_x)

#     return net 


def single_res_block(net, strides, kernel_size, depth):
    net = conv_layer_with_relu(net=net, num_filters = depth, kernel_size = [1,1], strides = strides)

    net = conv_layer_with_relu(net=net, num_filters = depth, kernel_size = kernel_size, strides = 1)
    return net

def parallel_resnet_blocks(net, strides, kernel_size, depth, cardinality):
    split_net = net
    parallel_res_block_list = list()
    for each_block in range(cardinality):
        parallel_res_block_list.append(single_res_block(net=split_net, strides=strides, kernel_size=kernel_size, depth=depth))
    concated_parallel_res_blocks=tf.concat(parallel_res_block_list, axis=3)
    return concated_parallel_res_blocks

def transition_layer(net, out_dim):
    strides = 1
    transition_layer = conv_layer_without_relu(net=net, num_filters = out_dim, kernel_size = 1,strides=strides)
    return transition_layer

def single_resneXt_block(net, strides, kernel_size, depth, cardinality):
    original_block = net
    out_dim = int(np.shape(original_block)[-1])
    parallel_resnet_blocks_out=parallel_resnet_blocks(net=net, strides=strides, kernel_size=kernel_size, depth=depth, cardinality=cardinality)
    out_transition_layer = transition_layer(net=parallel_resnet_blocks_out, out_dim=out_dim)
    final_output = tf.nn.relu(original_block + out_transition_layer)
    return final_output


def MY_ResneXt(Img, ImageSize, MiniBatchSize):
    
    num_classes = 10
    depth = 8
    cardinality = 32
    blocks = 3


    net = Img
    net = conv_layer_with_relu(net=net, num_filters = 32, kernel_size = 3, strides = 1)
        
    net = single_resneXt_block(net=net, strides=1, kernel_size = 3, depth=depth, cardinality=cardinality)
    net = conv_layer_with_relu(net=net, num_filters = 128, kernel_size = 1, strides=2)
    cardinality = 16
    net = single_resneXt_block(net=net, strides=1, kernel_size = 3, depth=depth, cardinality=cardinality)
    
    net = global_avg_pool(net, name='Global_avg_pooling');

    net = tf.contrib.layers.flatten(net)

    # net = tf.layers.dense(inputs=net, name='layer_fc1',
    #                   units=128, activation=tf.nn.relu)
    #Contruct Fully-connected layer
    net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)

    prLogits = net
    prSoftMax = tf.nn.softmax(net)
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

