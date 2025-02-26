# TinyGrad

A minimalistic implementation of automatic differentiation and neural networks from scratch.

## Overview
This project demonstrates core deep learning concepts:
- **Automatic differentiation** via computational graphs
- **Neural network building blocks** (layers, neurons, activations)
- **Gradient descent** optimization

## Features
- Scalar `Value` class with:
  - Basic operations (`+`, `*`, `/`, `**`)
  - Activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU, ELU)
  - Backpropagation algorithm
- Modular neural network components:
  - `Neuron` with configurable non-linearities
  - `Layer` for dense connections
  - `MLP` for multi-layer perceptrons
