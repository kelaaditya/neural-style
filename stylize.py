import numpy as np
import tensorflow as tf
import download_vgg
import vgg

from PIL import Image, ImageOps


VGG_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG = 'imagenet-vgg-verydeep-19.mat'

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

'''See GitHub gist: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
"The input images should be zero-centered by mean pixel (rather than mean image) subtraction. 
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68]."
'''
MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

#The deeper the layer, the higher the weight
LAYER_WEIGHTS = (0.5, 1.0, 1.5, 3.0, 4.0)