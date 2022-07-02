---
id: 67cxfcgit2ikpwemctrwlay
title: Introduction
desc: ''
updated: 1656751902252
created: 1656748586102
---

## Introduction

At the heart of deep learning is a wonderful object call neural networks.

Neural networks vaguely mimic the process of how the brain operates, with neurons firing bits of information.

## Linear Boundaries

The goal of the algorithm is to have y hat resembling y as closely as possible which is exactly equivalent to finding the boundary line that keeps the blue points above it and red ones below it.

## Higher Dimensions

What if we have more data columns??

Our euation won't be a line in 2D but a plane in 3 dimensions with a similar equation as before.

If we have n columns, our data just leaps into n-dimesional space

We can imagine that the points are just things with n cooridnates called x_1, x_2, x_3 all the way upto x_n with our labels  being y then our boundaries are just n-1 dimesnional hyperplane which is a high dimensional equivalent of a line in 2D or a plane in 3D

## Perceptrons

PERCEPTRON: The Building block of neural networks and its just an encoding of our equation into a small graph.

The Nodes multiplies the values coming from the nodes by values from the corresponding edges and finally adds them.

Note that we are using an implicit function here called the step function, the step functon returns 1 if input is positive otherwise it returns a zero.

So in reality these perceptrons can be seen as a combination of nodes where the first node calculates a linear equation on the inputs and the weights and the second node applies the step function to the result.

The summation sign represents a linear function in the first node and the drawing represents a step function in the second node. In the future we will use different step functions, so this is why it's useful to specify it in the node

Two ways to represent Perceptrons

One has bias unit coming from an input node with a value of one and the other has the bias inside the node.

## Why Neural Networks

Why are these objects called neural networks??

Because Perceptrons kinda look like neurons in the brain.

Perceptron calculates some quations on the input and decides to return a one or zero.

In a similar way neurons in the brain take inputs coming from the dendrites. These inputs are called nervous impulses. So what the neuron does is it does something with the nervous impulses and then it decides if it outputs a nervous impulse or not through the axon.

We ll create neural networks by concatenating these perceptrons so we will be mimicking the way the brain connects neurons by taking the output from one and turning it into an input for another one.

## Perceptrons as Logical Operators

we'll see one of the many great applications of perceptrons. As logical operators! You'll have the chance to create the perceptrons for the most common of these, the AND, OR, and NOT operators. And then, we'll see what to do about the elusive XOR operator.

## Error Functions

Error Function: Tells us how far are we from the solution.
Error is telling us badly we are doing at the moment and how far are we from an ideal solution.

In order for us to do gradient descent, our error function can not be discrete, it should be continuous and differentiable.

## Discrete vs Continuous

continuous error functions are better than discrete error functions, when it comes to optimizing. For this, we need to switch from discrete to continuous predictions.

The prediction is the answer we get from the algorithm.

The probability is a function of the distance from the line.

The way we move from discrete to continuous predictions is to simply change the activation function from the step function to the sigmoid function.

If you look at the plot for a sigmoid function, the probability would always be 50% if the score evaluates to 0.

## Softmax

Multi-Class Classification and Softmax

softmax function, which is the equivalent of the sigmoid activation function, but when the problem has 3 or more classes

Exponential function turns every number into a positive number.

## One-Hot Encoding

All our algorithms are numerical, so we need to input numbers.

You cannot do 0,1,2 because it will assume dependencies betwwen the classes we can't have. So we come up with one variable for each of the classes.
We may have more colums of data but at least there are no unnecessary dependencies.

## Maximum Likelihood

The best model would be more likely the one that gives the higher probabilities to the events that happeneed to us. We pick the model that gives the highest probability to the existing labels.

We want to calculate that the four points are of the colors that they actually are. if we assume that the colors of the points are independent events then the probability for the whole arrangement is the product of the probabilities of the four points.

What we mean by this is that if the model is given by these probability spaces then the probability that the points are of these colors is 0.0084.

P(all) is the product of all independent probabilities of each point. Therefore it helps indicate how well the model performs in classifying all the points

## Maximizing Probabilities

A better model will give us  a better probability.

What function turns products into sums?
log(ab) = log(a) + log(b), which is exactly what we need.

## Cross Entropy

Logarithm of a number betwwen 0 and 1 is always negative number since the logarithm of one is zero. We will take negative of the logarithm of the probilities and we will get positive numbers. Its called the cross entropy. A bad model has high cross entropy and a good model has low cross entropy.

We can think of the negatives of these logarithms as errors at each point.
Points that are correctly classified will have small errors and points that are mis-classified will have large errors.

Conclusion: Cross entropy will tell us if a model is good or bad.

The goal has changed from Maximizing a probability to minizing a cross entropy.

Error function is the cross entropy.

here's definitely a connection between probabilities and error functions, and it's called Cross-Entropy.

Cross entropy says If i have a bunch of events and probabilities, how likely is that those events happen based on the probabilities.
If its very likely then we have a small cross entropy otherwise a large cross entropy.

## Multi class entropy

cross-entropy is inversely proportional to the total probability of an outcome.

## Logistic regression

we're finally ready for one of the most popular and useful algorithms in Machine Learning, and the building block of all that constitutes Deep Learning. The Logistic Regression Algorithm. And it basically goes like this:

Take your data
Pick a random model
Calculate the error
Minimize the error, and obtain a better model

## Gradient descent

We will take the negative of the gradient of the error function at that point.

 If a point is well classified, we will get a small gradient. And if it's poorly classified, the gradient will be quite large.

So, a small gradient means we'll change our coordinates by a little bit, and a large gradient means we'll change our coordinates by a lot.

If this sounds anything like the perceptron algorithm, this is no coincidence!

Each point is adding a multiple of itself into the weights of the line in order to get the line to move closer towards it if its misclassified.
