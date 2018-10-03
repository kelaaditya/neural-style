import os
import sys
from tqdm import tqdm
import requests


def download_vgg(link, file_name):
    '''Download pre-trained VGGNet
    Checks if downloaded previously
    '''

    base_folder = os.path.dirname(__file__)
    file_name = os.path.join(base_folder, file_name)
    file_size = 534904783
    
    if os.path.exists(file_name) and os.stat(file_name).st_size == file_size:
        print('VGGNet ready')
    else:
        print('Downloading VGG... ')

        response = requests.get(link, stream=True)
        response.raise_for_status()

        block_size = 32 * 1024
        with open(file_name, 'wb') as data_file:
            with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        data_file.write(chunk)
                        pbar.update(len(chunk))
        print(f'{file_name} download complete.')


if __name__=="__main__":
    vgg_link = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
    vgg_filename = 'imagenet-vgg-verydeep-19.mat'
    try:
        download_vgg(vgg_link, vgg_filename)
    except Exception as e:
        print(e)
