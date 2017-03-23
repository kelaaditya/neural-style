'''
The paper by Gatys suggests that average pooling does better than max pooling
We default to average pooling but provide max pooling as the other option
We use the weights and biases of the already trained 19 layered VGGNet
'''


import numpy as np
import tensorflow as tf
import scipy.io

from PIL import Image


VGG19_LAYERS = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
               )


def _conv_layer(input_layer, weights, bias):
    conv_layer = tf.nn.conv2d(input_layer, 
                              filter=tf.constant(weights, name='weights'),
                              strides=(1, 1, 1,1),
                              padding='SAME')
    return(tf.nn.bias_add(conv_layer, bias))


def _relu_layer(input_layer):
    return(tf.nn.relu(input_layer))


def _pool_layer(input_layer, pool_func='avg'):
    '''pool_func has two options:
    'avg': Average pooling
    Else : Max pooling
    '''
    
    if pool_func == 'avg':
        return(tf.nn.avg_pool(input_layer,
                              ksize=(1, 2, 2, 1),
                              strides=(1, 2, 2, 1),
                              padding='SAME')
              )
    else:
        return(tf.nn.max_pool(input_layer,
                              ksize=(1, 2, 2, 1),
                              strides=(1, 2, 2, 1),
                              padding='SAME')
              )


def load_vgg(path, input_image, pooling_func='avg'):
    '''Loads VGGNet parameters
    
    Input:
        path: path to VGGNet
        input_image: input image
        pooling_func: Two options
                      'avg': average pooling 
                      Else : max pooling
    
    Output:
        graph: dictonary that holds the convolutional layers
    '''
    
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']
    graph = {}
    current_layer = input_image
    
    for i, layer in enumerate(VGG19_LAYERS):
        if layer[:4] == 'conv':
            weights, bias = vgg_layers[0][i][0][0][2][0]
            bias = bias.reshape(bias.size)
            current_layer = _conv_layer(current_layer, weights, bias)
        elif layer[:4] == 'relu':
            current_layer = _relu_layer(current_layer)
        elif layer[:4] == 'pool':
            current_layer = _pool_layer(current_layer, pool_func='avg')
        graph[layer] = current_layer
    
    return(graph)

