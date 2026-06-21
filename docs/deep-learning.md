# Project: Basic Deep Learning building blocks from scratch

Idea contributed by Maximilian Gollwitzer <maximilian.gollwitzer@tum.de>.

**Warning:** *This project may be particularly difficult. This project requires some knowledge on the technical details
of neural networks (backpropagation) and the underlying mathematical concepts (matrix calculus). For someone unfamiliar 
with the topic this entails reading additional time to acquire the required knowledge.*

## Overview

The goal of this project is to implement the basic building blocks of Neural Networks in C++ using
only a library for matrix computations (e.g. Eigen).

## Content

The project should include a Dataloader, a few simple NN layers, a class for generating parameterised
NN architectures from the implemented layers and an optimiser to update the network weights using the individual layers gradients.

*(See next section if some terms are unfamiliar)*

## Recommended Knowledge / Sources

A basic understading of neural networks and backpropagation (Or: what do we need gradients for and how do we update weights?): https://www.youtube.com/watch?v=aircAruvnKk

Some matrix calculus: https://explained.ai/matrix-calculus/

PyTorch docs explains typical components/layers quite well imo: https://pytorch.org/docs/stable/nn.html

What is a dataloader: https://medium.com/unpackai/dataloaders-in-ai-1bc1e2ea94b3 or https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
(Note that our dataloader should really just load image files from disk into memory)

What is an optimiser: https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/

## Sprint 1: Dataloader and Linear Layer

This sprint is about setting up the project, making data accessible in the code and performing basic computations.

*Note: Dataloaders typically offer some advanced functions like applying transforms and work for various dataset structures.
In our case we really just want it to load one specific dataset from disk into some usable format.*

### Sprint 1: Definition of "done"

- The Dataloader should load .png images from the disk into memory in some usable format.
- Functions for computing the linear layer forward and backward (gradient) pass are implemented.
- Functions for computing sigmoid forward and backward (gradient) pass are implemented. 


## Sprint 2 Modularising using OOP

This sprint is about structuring the code in a modular fashion to ultimately allow flexible generation of network architectures.

### Sprint 2: Definition of "done"
The following functionalities are implemented:
- A Layer parent class offering methods for forward and backward pass and the following child classes:
  - Linear layer, parameterised by the number of input featuers and the number of neurons (output features)
  - Sigmoid layer
  - Softmax & Cross Entropy Loss (*)
- A Network class: Accepts a vector of layers, an optimiser, a loss layer and supports:
  - forward: computes the output of the layers forward functions applied in sequential order
  - backward: all layers store their gradients w.r.t. the loss function
  - update: all layers weights are updated by a step of the specified optimiser
- An optimiser parent class offering a step method and the following child class:
  - Vanilla SGD, accepting a learning rate parameter and performing a simple gradient descent update

*Note: While these are a lot of components, all of them should be implementable in less than 25, most in less than 10 lines of code.*

*(\*) It is much easier to implement the CE Loss backward pass (gradient) assuming that its inputs come from the Softmax function. I recommend putting the gradient through Softmax and CE into the CE backward function and making the Softmax backward pass the identity (Those are always used in combination). The explanation for why is given here: https://www.parasdahal.com/softmax-crossentropy*
## Sprint 3 Code quality and performance

This sprint is about finalising the project by cleaning and optimising the code.

### Sprint 3: Definition of "done"

[ *performance* ]:
- Matrix operations are performed by an optimised library where possible
- Move operations are used where possible

[ *features* ]:
- A more performant optimiser such as SGD + Momentum or ADAM is implemented.
- A second loss function, e.g. MSE loss is implemented
- The Parameterised Leaky ReLu, parameterised by the slope for negative inputs, activation function is implemented

[ *code structure* ]:
- The dataloader uses iterators

[ *style points* ]:
- Achieve at least 30% accuracy on Mnist ;)
