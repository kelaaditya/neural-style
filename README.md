# neural-style
A basic TensorFlow implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) 
by Gatys, Ecker and Bethge.  


## Example
The program `stylize.py` was run with 300 iterations (default value) to get the following result. The content loss weight parameter was choosen to be 0.1
![neural_style_image](https://github.com/kelaaditya/neural-style/blob/master/neural_styled/stylized_koeln_cathedral.jpg "content loss weight = 0.1")

The content image was a night-time image of the Cologne Cathedral:
![content_image](https://github.com/kelaaditya/neural-style/blob/master/content/koeln_cathedral.jpg "Cologne Cathedral")

The style image was a famous Escher graphic:  
<img src="https://github.com/kelaaditya/neural-style/blob/master/style/escher_birds.jpg" width="400" height="400" />


## Running
(An argument parser is coming soon. Until then.. )  

`python stylize.py`

`stylize.py` will download the VGG19 data and load the weights and biases. It imports from `download_vgg.py` to download the data and `vgg.py` to create a dictionary of each layer.

The default number of iterations is 300.



## References
Based on the material from the Stanford course [TensorFlow for Deep Learning Research](http://web.stanford.edu/class/cs20si/)
