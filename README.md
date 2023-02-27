# Neural-Image-Generator

## Introduction
Neural Transfer is a popular technique used in computer vision and deep learning to generate images that combine the style of one image with the content of another. This is achieved by training a neural network to extract the style and content information from two input images, and then using that network to generate a new image that combines the two.

The Neural Transfer by PyTorch project is a Python-based implementation of this technique using the PyTorch deep learning framework. It provides a simple and flexible way to perform Neural Transfer on any pair of images, allowing users to experiment with different combinations of styles and content.

The project is built around a pre-trained deep neural network architecture called VGG-19, which is used to extract the feature maps from the input images. These feature maps are then used to calculate the content loss and style loss, which are combined to form the total loss function. The optimization process then aims to minimize this loss function to generate the final output image.

The Neural Transfer by PyTorch project provides a user-friendly interface that allows users to adjust various parameters and settings, such as the learning rate, number of iterations, and style and content weights, to fine-tune the generated images. Additionally, the project includes a set of pre-trained models for different styles, making it easy for users to quickly generate images with a specific style.

Overall, the Neural Transfer by PyTorch project is a powerful and flexible tool for generating artistic images that combine the style and content of two input images. It is a great example of how deep learning can be used to create novel and visually stunning images with relatively little effort.

## Data
Style Image: Sunflower, by Van Gogh

https://philamuseum.org/collection/object/59202

Contene Image: My cat image

## Result
![output](https://user-images.githubusercontent.com/90078254/221447249-3be98de2-bf4b-47d1-8367-c852dc00dc65.jpg)


## Reference
NEURAL TRANSFER USING PYTORCH: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

Author: Alexis Jacq; Edited by: Winston Herring


Neural Style Transfer Using PyTorch: https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa

Author: Aman Kumar Mallik
