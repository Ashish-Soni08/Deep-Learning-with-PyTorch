---
id: 67cxfcgit2ikpwemctrwlay
title: IntroductiontoNNs
desc: ''
updated: 1656801717150
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

## Perceptron vs Gradient Descent

In Perceptron algorithms not every point changes weights only the misclassified ones.
If a point is correctly classified, do nothing. but in Gradient Descent algorithms, the point tells the line to go farther away.

## Non Linear Data

Datasets requires highly non-linear boundaries. This is where neural networks can show their full potential.

Non-Linear Models - Data is not seperable by a line we are going to create a probability function, which will not be linear.

## Neural Network Architecture

we're ready to put these building blocks together, and build great Neural Networks! (Or Multi-Layer Perceptrons, however you prefer to call them.)

This first two videos will show us how to combine two perceptrons into a third, more complicated one

How to create these non-linear models?
We are going to combine two linear models into a nonlinear model as follows: Its almost like we are doing arithmetic on models like saying this line + this line equals that curve.

Linear models as we know is a whole probability space.

Multiple layers

They can be way more complicated! In particular, we can do the following things:

Add more nodes to the input, hidden, and output layers.
Add more layers.

Neural Networks have a certain Special Architecture with layers. The first layer is called the input layer. The next layer is called the hidden layer which is a set of linear models created with the first input layer and the final layer is called the output layer where linear models are combined to obtain a non-linear model.

You can different acrhitectures.

If we have more layers then we have a deep neural network, our linear models combine to create non-linear models that further combine to create even more nonlinear models.
That neual network will just split the n-dimensional space with a highly non-linear boundary.

To classify 26 letters you will need 26 nodes in the output layer. (Alternatively, 52 nodes for 26 uppercase and 26 lowercase letters)

## Feedforward

Feedforward is the process neural networks use to turn the input into an output.

How to train neural networks?
Training them really means what parameters should they have on the edges in order to model our data well.

Neural networks take the input vector and then apply a sequence of linear models and sigmoid functions. These maps when combined become a highly non-linear map.
This is the feedforward process that the neural networks use to obtain the prediction from the input vector.

Error Function
Just as before, neural networks will produce an error function, which at the end, is what we'll be minimizing.

## Backpropagation

backpropagation. In a nutshell, backpropagation will consist of:

Doing a feedforward operation.
Comparing the output of the model with the desired output.
Calculating the error.
Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
Use this to update the weights, and get a better model.
Continue this until we have a model that is good.

Gradient is simply the vector formed by all the partial derivatives of the error function with respect to the weights w1 to wn and the bias b.

We calculate the gradient of the error function which is like asking the point what does it want the model to do?

Chain Rule: When composing functions the derivatives just multiply.
Feedforwarding is literally composing a bunch of functions and backpropagation is literally taking the derivative at each piece. since taking the derivative of a composition is the same as multiplying the partial derivative, so we all do is multiply a bunch of partial derivatives to get what we want.

## Training Optimization

There are many things that can fail

- Architecture can be poorly chosen
- Our data can be noisy
- Model is taking too long to train

## Testing

In ML, if we have to choose between a simpler model that does the job and a complex model that does the job a little bit better, we are going to choose the simple model.

## Overfitting and Underfitting

Whats the problem with trying to Godzilla with a flyswatter? that we are oversimplyfying the problem, we are trying a solution that is too simple and won't do the job, its called under-fitting. Sometimess referred as Error due to bias.

Killing a fly with a bazooka, its overly complicated and it will lead to bad solutions and extra complexity when we can use a much simpler solution instead. This is called overfitting. Fit the data well but fail to generalize. Sometimes referred to as error due to variance.

## Early Stopping

We do gradient descent until the testing error stops decreasing and starts to increase. At that moment we stop. This algorithm is called early stopping and is widely used to train neural networks.

## Regularization

 Remember that the error is smaller if the prediction is closer to the actual label

Large Coefficients -> overfitting

Penalize larger weights

Tweak the error function - add a term which is big when the weights are big. Two ways:

- add the sums of absolute values of the weights times a constant lambda. L1 regularization -> when we apply L1 we tend to end up with sparse vectors, that means small weights will tend to go to zero. So if we want to reduce the number of weights and end up with a small set, we can use L1 for good feature selection.
- add the sum of the squares of the weights times that same constant. L2 regularization -> tries to maintain all the weights homogeneously small. Normally gives better results for training models.
- The lambda parameter will tell us how much we want to penalize the cofficients. If lambda is large we penalze them a lot.

## Dropout

Sometimes one part of the neural network has very large weights and ends up dominating all the training. So we randomly turn off some nodes.

## Local Minima

Stuck in a Local Minimum

## Random restart

We start from a few different random places and do gardient descent from all of them. This increases the probability that we will get to the global minimum or atleast a pretty good local minimum.

## Vanishing Gradient

Derivative is almost zero is not good since it tells in which direction to move. 

This is worse in Linear Perceptrons

The best way to fix it is to change the activation function - Hyperbolic tangent function -> range -1 and 1, Relu function ->  x and 0

## Batch and Stochastic Gradient Descent

If we many, many data points, these are huge matrix computations that use tons and tons of memory and all that for just a single step. More steps will need a long time and a lot of computing power.

Do we need to Plug in all our data every time we take a step?

If the data is well distributed, it would give us a very good idea of what the gradient would be.

The idea behind stochastic gradient descent is simply that we take small subsets of data, run them through the network, calculate the gradient of the error function based on those points, and then move one step in that direction. We split the data into several batches.

## Learning Rate decay

If your model is not working decrease the learning rate. The best learning rates decrease as the model is getting closer to a solution.
Rule for decreasing learning rate:

- If steep: long steps
- If plain: short steps

## Momentum - another way to solve a local minimum problem

The idea is to walk a bit fast with momentum and determination in a way that if you get stuck in a local minimum, you can sort of power through and get over the hump to look for a lower minimum.

Momentum is a constant beta betwwen 0 and 1 that attaches to the step as follows: the previous step gets multiplietd by 1, the one before by beta, then beta square and so on.
