"""
TODO: Argument Parser
"""

""" 
A basic TensorFlow implementation of the paper 
"A Neural Algorithm of Artistic Style" - Gatys et al.

Based on the Tensorflow for Deep Learning Research course at Stanford.
More details here: http://web.stanford.edu/class/cs20si/index.html
"""




import numpy as np
import tensorflow as tf
import scipy.misc
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


def noisy_image(given_image, image_height, image_width, noise_ratio=0.5):
    image = np.random.uniform(-10, 10, (1, image_height, image_width, 3)).astype(np.float32)
    noise_image = image * noise_ratio + given_image * (1 - noise_ratio)
    return(noise_image)


def _gram_matrix(T, n, m):
    T = tf.reshape(T, (m, n))
    gram_matrix = tf.matmul(tf.transpose(T), T)
    return(gram_matrix)


def content_loss(P, F):
    return(tf.reduce_sum((F - P) ** 2) / (4.0 * P.size))


def style_loss(A, G):
    '''Calculates the layer loss
    
    Input:
        A: feature map of the real image
        G: feature map of the generated image
    '''
    total_filters = A.shape[-1]
    map_size = A.shape[1] * A.shape[2]
    G_A = _gram_matrix(A, total_filters, map_size)
    G_G = _gram_matrix(G, total_filters, map_size)
    loss = tf.reduce_sum((G_A - G_G)**2 / ((2 * total_filters * map_size) **2))
    return(loss)


def total_style_loss(feature_layers, graph):
    loss_list = [style_loss(feature_layers[i], graph[STYLE_LAYERS[i]]) for i in range(len(STYLE_LAYERS))]
    total_loss = np.sum([LAYER_WEIGHTS[j] * loss_list[j] for j in range(len(STYLE_LAYERS))])
    return(total_loss)


def combined_loss_function(graph, input_image, style_image, content_image, content_loss_weight, style_loss_weight):
    with tf.Session() as sess:
        sess.run(input_image.assign(content_image))
        a = sess.run(graph[CONTENT_LAYER])
    content_loss_value = content_loss(a, graph[CONTENT_LAYER])
    
    with tf.Session() as sess:
        sess.run(input_image.assign(style_image))
        A = sess.run([graph[i] for i in STYLE_LAYERS])
    style_loss_value = total_style_loss(A, graph)
    
    total_loss_value = content_loss_weight * content_loss_value + style_loss_weight * style_loss_value
    
    return(content_loss_value, style_loss_value, total_loss_value)


def stylize(content_image, 
            style_image, 
            content_loss_weight, 
            style_loss_weight, 
            initial_image_noise_ratio, 
            pooling_func='avg'):
    
    content_image_width, content_image_height = content_image.size
    
    download_vgg.download_vgg(VGG_LINK, VGG)
    
    input_image = tf.Variable(np.zeros([1, content_image_height, content_image_width, 3]), dtype=tf.float32)
    graph = vgg.load_vgg('./imagenet-vgg-verydeep-19.mat', input_image, pooling_func)
    
    content_image = content_image - MEAN_PIXELS
    
    style_image = style_image - MEAN_PIXELS
    
    content_loss, style_loss, total_loss = combined_loss_function(graph, 
                                                                  input_image, 
                                                                  style_image, 
                                                                  content_image, 
                                                                  content_loss_weight, 
                                                                  style_loss_weight)
    
    optimizer = tf.train.AdamOptimizer(1.0).minimize(total_loss)
    
    initial_image = noisy_image(content_image, content_image_height, content_image_width, initial_image_noise_ratio)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(input_image.assign(initial_image))
        for iterations in range(300):
            sess.run(optimizer)
        generated_image, total_loss_value = sess.run([input_image, total_loss])
        
    generated_image = generated_image + MEAN_PIXELS        
    final_image = np.clip(generated_image[0], 0, 255).astype('uint8')
    scipy.misc.imsave('./neural_styled/stylized_koeln_cathedral.jpg', final_image)
    print('Stylized image saved')
    

if __name__ == "__main__":
    content_image = Image.open('./content/koeln_cathedral.jpg')
    refit_content_image = ImageOps.fit(content_image, content_image.size, Image.ANTIALIAS)

    #Note that we fit style image to the shape of the content image
    style_image = Image.open('./style/escher_birds.jpg')
    refit_style_image = ImageOps.fit(style_image, content_image.size, Image.ANTIALIAS)
    
    stylize(refit_content_image, 
            refit_style_image, 
            content_loss_weight=0.1, 
            style_loss_weight=1, 
            initial_image_noise_ratio=0.5, 
            pooling_func='avg')
    
