# neural-style
A basic TensorFlow implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) 
by Gatys, Ecker and Bethge.  


## Example
The program `stylize.py` was run with 300 iterations (default value) and average pooling to get the following result. The content loss weight parameter was choosen to be 0.1
![neural_style_image](https://github.com/kelaaditya/neural-style/blob/master/neural_styled/stylized_koeln_cathedral.jpg "content loss weight = 0.1")

The content image was a night-time image of the Cologne Cathedral:
![content_image](https://github.com/kelaaditya/neural-style/blob/master/content/koeln_cathedral.jpg "Cologne Cathedral")

The style image was a famous Escher graphic:  
<img src="https://github.com/kelaaditya/neural-style/blob/master/style/escher_birds.jpg" width="400" height="400" />

## Example 2
This image was the result of 1000 iterations and average pooling but with a content loss weight of 0.2:
![neural_style_image_2](https://github.com/kelaaditya/neural-style/blob/master/neural_styled/stylized_koeln_cathedral_starry.jpg)

The style image is The Starry Night by Gogh:  
<img src="https://github.com/kelaaditya/neural-style/blob/master/style/starry_night.jpg" width="480" height="400" />

## Running

`python stylize.py`

`stylize.py` will download the VGG19 data and load the weights and biases. It imports from `download_vgg.py` to download the data and `vgg.py` to create a dictionary of each layer.

The default number of iterations is 300.


## Implementation Details
Images are optimized using the ADAM optimizer instead of L-BFGS.

Style losses are calculated over the `conv1_1, conv2_1, conv3_1, conv4_1` and `conv5_1` layers and the content loss is calculated using the `conv4_2` layer. 

Average pooling was used instead of max-pooling

The input images were zero centered using MEAN_PIXEL subtraction.  
See this for more information: https://gist.github.com/ksimonyan/211839e770f7b538e2d8


## References
Based on the material from the Stanford course [TensorFlow for Deep Learning Research](http://web.stanford.edu/class/cs20si/)
