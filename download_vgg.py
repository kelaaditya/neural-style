import os
import sys
import numpy as np
import scipy.misc
import urllib.request

from PIL import Image, ImageOps


VGG = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

def download_vgg(link, file_name):
    '''Download pre-trained VGGNet
    Checks if downloaded previously
    '''
    
    def report_hook(block_num, block_size, total_size):
        '''A report hook for urlretrieve
        '''
        current_size = block_num * block_size
        if total_size > 0:
            percent_size = current_size / total_size * 100
            progress_string = '{0}, {1} out of {2}\n'.format(percent_size, current_size, total_size)
            if percent_size % 10 == 0:
                sys.stdout.write(progress_string)
    
    if os.path.exists(file_name) and os.stat(file_name).st_size == 534904783:
        print('VGGNet ready')
    else:
        urllib.request.urlretrieve(link, file_name, report_hook)

download_vgg(VGG, VGG_MODEL)