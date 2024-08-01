# Variational Autoencoder

## Introduction
This is a simple implementation of a Variational Autoencoder (VAE) in PyTorch. The VAE is a generative model that learns to encode and decode data, which is trained to encode these images into a lower-dimensional latent space and then decode them back to the original image.

## Results

### MNIST

![MNIST](https://github.com/aakashvardhan/s22-variational-autoencoders/blob/main/asset/wrong_label_images_mnist.png)


### CIFAR-10

![CIFAR-10](https://github.com/aakashvardhan/s22-variational-autoencoders/blob/main/asset/wrong_label_images_cifar10.png)

## Usage

### Training

To train the model, run the following command:

```bash
# change the dataset and epochs as per your requirement
# "mnist" or "cifar10"
python trainer.py --dataset "mnist" --epochs 10
```