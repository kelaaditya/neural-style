import os
import sys
import numpy as np
import scipy.misc
import urllib

from PIL import Image, ImageOps


VGG = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

def download_vgg(link, file_name):
    '''Download pre-trained VGGNet
    Checks if downloaded previously
    '''
    
    if os.path.exists(file_name) and os.stat(file_name).st_size == 534904783
        print('VGGNet ready')
    else:
        urllib.request.urlretrieve(link, file_name, report_hook)