# Neural-Network-with-TensorSlow
This repository contains my work on TensorSlow, an educational deep learning framework inspired by TensorFlow. 
The project was developed as part of the university course Build Your Own Neural Network.

The goal was not only to train neural networks, but also to understand how modern deep learning frameworks work 
internally by implementing important components from scratch, including automatic differentiation, computational graphs,
layers, activation functions, and optimization algorithms.

# Vision Transformer Extension
In the Vision Transformer branch I extended TensorSlow with a Vision Transformer (ViT) based on the paper:
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
The implementation adapts the original Vision Transformer architecture to the TensorSlow framework and demonstrates how 
Transformer-based image classification can be built without relying on PyTorch or TensorFlow. The model was trained and 
evaluated on the CIFAR-10 dataset.


# Important files in the Vision Transformer Branch

##  1. vit_tensorslow.py
This is the main entry point of the Vision Transformer implementation.
It contains: 
- The complete training loop
- Accuracy and loss evaluation
- Hyperparameter configuration
- CIFAR-10 data loading
- Plot generation

It also contains the following core classes:
#### PatchEmbedding
Converts images into patch embeddings and adds positional information and a CLS token.
#### TransformerMLP
Implements the feed-forward network used inside each Transformer encoder block. 
#### TransformerEncoder
Combines Layer Normalization, Self-Attention, residual connections and the MLP block.
#### VisionTransformer
The complete Vision Transformer architecture used for image classification.


##  2. model.py 
Contains the implementation of: 

####  SelfAttention

A custom implementation of scaled dot-product attention.
The attention mechanism computes Query, Key and Value matrices and performs attention scoring without using external 
deep learning libraries.

## 3. layers.py

Contains several neural network layers including:

- Linear
- LayerNorm
- Dropout
- Sequential

The custom LayerNorm implementation was added to support Transformer architectures.


## 4. functions.py

Contains many mathematical operations used by the autograd engine.

Additional functions implemented for the Vision Transformer include:

- GeLU activation
- BatchMatMul
- MatMul
- Concat
- GetItem
- Softmax support for attention

These operations required custom backward implementations so gradients could correctly propagate 
through the computational graph.

## Technical Challenges

During development several non-trivial problems had to be solved:

- Object dtype errors during optimization
- Gradient propagation failures
- Computational graph breaks caused by .data
- Broadcasting issues
- Concatenation backpropagation
- Matrix multiplication gradients
- Weak reference debugging
- Transformer-specific tensor shape handling

A significant part of the project involved debugging and extending TensorSlow's automatic differentiation engine to 
support the Vision Transformer architecture.


## Results

The final Vision Transformer achieved approximately 58% test accuracy on CIFAR-10 after 10 training epochs on 
CPU-only hardware.

While this is below state-of-the-art performance, the primary goal of the project was educational: understanding how 
Transformer architectures and deep learning frameworks work internally by implementing them from scratch.

***

## Skills Demonstrated
- Deep Learning Fundamentals
- Automatic Differentiation
- Backpropagation
- Transformer Architectures
- Vision Transformers (ViT)
- Neural Network Training
- Python
- NumPy
- Software Debugging
- Machine Learning Engineering
- Scientific Computing