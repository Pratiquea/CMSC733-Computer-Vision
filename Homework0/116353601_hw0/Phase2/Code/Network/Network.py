"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

# def new_weights(shape):
#     return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
# def new_biases(length):
#     return tf.Variable(tf.constant(0.05, shape=[length]))


# def new_conv_layer(input,              # The previous layer.
#                    num_input_channels, # Num. channels in prev. layer.
#                    filter_size,        # Width and height of each filter.
#                    num_filters,        # Number of filters.
#                    use_pooling=True):  # Use 2x2 max-pooling.

#     # Shape of the filter-weights for the convolution.
#     # This format is determined by the TensorFlow API.
#     shape = [filter_size, filter_size, num_input_channels, num_filters]

#     # Create new weights aka. filters with the given shape.
#     weights = new_weights(shape=shape)

#     # new_weights(shape=shape)

#     # Create new biases, one for each filter.
#     biases =  new_biases(length=num_filters)
#     # biases = new_biases(length=num_filters)

#     # But e.g. strides=[1, 2, 2, 1] would mean that the filter
#     # is moved 2 pixels across the x- and y-axis of the image.
#     # The padding is set to 'SAME' which means the input image
#     # is padded with zeroes so the size of the output is the same.
#     layer = tf.nn.conv2d(input=input,
#                          filter=weights,
#                          strides=[1, 1, 1, 1],
#                          padding='SAME')

#     # Add the biases to the results of the convolution.
#     # A bias-value is added to each filter-channel.
#     layer += biases

#     # Use pooling to down-sample the image resolution?
#     if use_pooling:
#         # This is 2x2 max-pooling, which means that we
#         # consider 2x2 windows and select the largest value
#         # in each window. Then we move 2 pixels to the next window.
#         layer = tf.nn.max_pool(value=layer,
#                                ksize=[1, 2, 2, 1],
#                                strides=[1, 2, 2, 1],
#                                padding='SAME')

#     # Rectified Linear Unit (ReLU).
#     # It calculates max(x, 0) for each input pixel x.
#     # This adds some non-linearity to the formula and allows us
#     # to learn more complicated functions.
#     layer = tf.nn.relu(layer)

#     # Note that ReLU is normally executed before the pooling,
#     # but since relu(max_pool(x)) == max_pool(relu(x)) we can
#     # save 75% of the relu-operations by max-pooling first.

#     # We return both the resulting layer and the filter-weights
#     # because we will plot the weights later.
#     return layer, weights

# # A function to convert the ouput of 2nd convolution layer to proper input for 1st neural layer 
# def flatten_layer(layer):
#     # Get the shape of the input layer.
#     layer_shape = layer.get_shape()

#     # The shape of the input layer is assumed to be:
#     # layer_shape == [num_images, img_height, img_width, num_channels]

#     # The number of features is: img_height * img_width * num_channels
#     # We can use a function from TensorFlow to calculate this.
#     num_features = layer_shape[1:4].num_elements()
    
#     # Reshape the layer to [num_images, num_features].
#     # Note that we just set the size of the second dimension
#     # to num_features and the size of the first dimension to -1
#     # which means the size in that dimension is calculated
#     # so the total size of the tensor is unchanged from the reshaping.
#     layer_flat = tf.reshape(layer, [-1, num_features])

#     # The shape of the flattened layer is now:
#     # [num_images, img_height * img_width * num_channels]

#     # Return both the flattened layer and the number of features.
#     return layer_flat, num_features


# def new_fc_layer(input,          # The previous layer.
#                  num_inputs,     # Num. inputs from prev. layer.
#                  num_outputs,    # Num. outputs.
#                  use_relu=True): # Use Rectified Linear Unit (ReLU)?

#     # Create new weights and biases.
#     weights = new_weights(shape=[num_inputs, num_outputs])
#     biases = new_biases(length=num_outputs)

#     # Calculate the layer as the matrix multiplication of
#     # the input and weights, and then add the bias-values.
#     layer = tf.matmul(input, weights) + biases

#     # Use ReLU?
#     if use_relu:
#         layer = tf.nn.relu(layer)

#     return layer

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
    # x_image = tf.reshape(Img, [-1, ImageSize[0], ImageSize[1], ImageSize[3]])

    #Convolution layer 1
    filter_size1 = 3      #filter size for first convolution layer(i.e. 5x5 pixels)
    num_filters1 = 8     #number of filters in first convolution layer

    # print()

    # #Convolution layer 2
    filter_size2 = 3      #filter size for second convolution layer(i.e. 5x5 pixels)
    num_filters2 = 128     #number of filters in second convolution layer

    # # Fully-connected layer.
    fc1_size = 128             # Number of neurons in fully-connected layer.
    fc2_size = 128             # Number of neurons in fully-connected layer.
    num_classes = 10

    # #Constructing fist convolution layer
    # layer_conv1, weights_conv1 = \
    # new_conv_layer(input=Img,
    #                num_input_channels=ImageSize[2],
    #                filter_size=filter_size1,
    #                num_filters=num_filters1,
    #                use_pooling=True)

    # layer_conv2, weights_conv2 = \
    # new_conv_layer(input=layer_conv1,
    #                num_input_channels=num_filters1,
    #                filter_size=filter_size2,
    #                num_filters=num_filters2,
    #                use_pooling=True)

    # layer_flat, num_features = flatten_layer(layer_conv2)

    # #Contruct 1st Fully-connected layer
    # layer_fc1 = new_fc_layer(input=layer_flat,
    #                      num_inputs=num_features,
    #                      num_outputs=fc1_size,
    #                      use_relu=True)


    # #Contruct 2nd Fully-connected layer
    # layer_fc2 = new_fc_layer(input=layer_fc1,
    #                      num_inputs=fc1_size,
    #                      num_outputs=fc2_size,
    #                      use_relu=True)

    # #Contruct 3rd Fully-connected layer
    # layer_fc3 = new_fc_layer(input=layer_fc2,
    #                      num_inputs=fc2_size,
    #                      num_outputs=num_classes,
    #                      use_relu=False)
    # #normalized probalities of logits
    
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

