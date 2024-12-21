# Introduction

The main task was to implement a backpropagation algorithm to train a
small perceptron neural network, with a maximum of 4 neurons in the
hidden layers, to solve the XOR problem. Specifically, the goal was to
test the model's training process using the backpropagation algorithm
for up to 500 epochs and a maximum learning rate of 0.1.

# Implementation in Code

## Introduction to Implementation

Since this is a backpropagation algorithm, it was necessary to implement
the `backward` method for all layers in the neural network. Essentially,
the backpropagation algorithm works by calculating the gradient of the
loss function with respect to each weight in the network and updating
these weights accordingly to minimize the loss. This process involves
two main steps:

1.  **Forward Pass**: Computing the output of the network by passing
    inputs through all layers sequentially.

2.  **Backward Pass**: Calculating gradients layer by layer, starting
    from the output layer and propagating them back to the input layer.
    Essentially, backward functions represent the derivatives of the
    corresponding forward functions.

The implementation ensures that each layer computes its own gradients
and passes the necessary information backward to the previous layer,
making the model inherently modular.

Representation of the data in code looks like this:

| X   | y_true |
| --- | ------ |
| 0 0 | 0      |
| 0 1 | 1      |
| 1 0 | 1      |
| 1 1 | 0      |

*XOR truth table representation in code*


Here, $X$ represents the table that will be fed into the neural network,
while $y\_true$ contains all the true labels corresponding to the truth
table.

## Tanh - activating function

This function provides the non-linear activation for neurons.
Essentially, when an input approaches $-\infty$, the output tends toward
$-1$, and when the input approaches $+\infty$, the output tends toward
$1$.

In math forward of Tanh looks like this:
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

    return self.output = np.tanh(x)

And backward looks like this: $$\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)$$

    return grad_output * (1 - self.output ** 2)

![Tanh](https://github.com/user-attachments/assets/e04bfa49-1818-47bb-b5bf-e68c6ef30dcd)

## Sigmoid - activation function

This function is an activation function, similar to ReLU or Tanh, but
sigmoid is more aggressive with logits. Specifically, this function
pushes logits less than -2 closer to 0. The main issue is that this
behavior can result in many zeros in the output, which in turn may lead
to numerous zeros in the gradients, preventing the weights from updating
effectively.

![Sigmoid](https://github.com/user-attachments/assets/342f5ba5-ed5d-4479-81c6-2b57a7778629)


In the code forward function of sigmoid is this:

    self.output = 1 / (1 + np.exp(-x))
    return self.output

And the backward is this:

    return grad_output * self.output * (1 - self.output)

## ReLU - Activation Function

The ReLU is also an activation function in neural networks. It provides
a simple and effective non-linear activation by zeroing out all negative
input values while leaving positive values unchanged. Specifically, for
any input $x$, the output of ReLU is defined as:

$$\text{ReLU}(x) = \max(0, x)$$

But because of this simplicity, ReLU may have the same problem as
sigmoid if we have too many numbers that are less than 0's.

![ReLU](https://github.com/user-attachments/assets/031a7e70-0cf3-4f73-a7b1-09d4a33483fe)


In the code, the forward function of ReLU is defined as:

    self.output = np.maximum(0, x)
    return self.output

And the backward function is the derivative of ReLU:

    return grad_output * (self.output > 0)

This derivative is simple: it returns 1 for positive input values and 0
for non-positive values, allowing gradients to propagate only through
activated neurons.

## MSE - loss function

In the code, the MSE is implemented as a class with three methods: the
constructor for initializing the object and two others, forward and
backward.

By definition, the forward method computes the Mean Squared Error (MSE)
using the formula:
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n \left( y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)} \right)^2$$
And in python it looks like this:

    return np.sum((y_pred - y_true) ** 2) / y_true.size

To calculate the MSE, we first compute the element-wise difference
between the predicted and true matrices, then square the differences,
sum them, and finally divide by the size of the matrix containing the
true values.

The backward method computes the gradient of the MSE with respect to the
predictions, following this formula:
$$\frac{\partial \text{MSE}}{\partial y_{\text{pred}}} = \frac{2}{n} \left( y_{\text{pred}} - y_{\text{true}} \right)$$
In python this is implemented as:

    return 2 * (self.y_pred - self.y_true) / self.y_true.size

This method is crucial in the backpropagation algorithm since it will
return how much the predicted values should be adjusted to minimize the
loss.

## Linear Layer

The linear layer is a fundamental component of a perceptron neural
network. It consists of a matrix of weights, a matrix of biases, and
matrices for gradients. During the forward pass, the linear layer
transforms the input matrix $X$ using the following formula:

$$y = W \cdot X + b$$

Here:

-   'y' is the output of the linear layer.

-   'X' is the input matrix.

-   'W' is the weights matrix, representing the coefficients applied to
    the input.

-   'b' is the biases matrix, which adds an offset to the output.

During backpropagation, gradients are computed and used to adjust the
weights and biases to minimize the error.

The linear layer contains the following components:

1.  **Matrix of weights:** This defines the $W$ in the equation. It acts
    as the linear coefficient for each element in the input matrix $X$.
    The weights matrix determines the connections between neurons in
    adjacent layers.

2.  **Matrix of biases**: This provides an offset ($b$) to the linear
    transformation, ensuring the network can model shifts in the data
    effectively.

3.  **Input matrix**: The data that is fed into the neural network. For
    instance, in image recognition, this could be a flattened array of
    pixel values (e.g., from a 28x28 image). In simpler tasks like XOR,
    it could be all possible combinations of 0 and 1.

4.  **Matrices of gradients for weights and biases**: These are critical
    for backpropagation. Gradients indicate how much each weight and
    bias needs to be adjusted to reduce the error. They are computed
    based on the loss function and the gradients propagated backward
    from subsequent layers.

# Testing with 1 hidden layer

The initial predictions of the model were not very close to the expected
output. The prediction was approximately 0.5 for every element of the
$X$ matrix. The first improvement was observed when I changed the
learning rate to 0.4, but since I wanted to stay within the boundaries
of our task (from 0.01 to 0.1), I focused on making optimizations and
testing different layer combinations.

The first step was to find the best layer configuration. For a single
hidden layer, the optimal setup is as follows:

    layers = [
        LinearLayer(2, 4),
        Tanh(),
        LinearLayer(4, 1),
        Sigmoid()
    ]

The average output for the XOR problem with this configuration was:

    With input [0 0] prediction = [0] and raw prediction = [0.25285789]
    With input [0 1] prediction = [0] and raw prediction = [0.43841484]
    With input [1 0] prediction = [1] and raw prediction = [0.65821009]
    With input [1 1] prediction = [0] and raw prediction = [0.48783318]

The output still appeared somewhat random. Increasing the learning rate
significantly improved the results, but since the maximum allowed rate
was 0.1, I explored further optimizations to stay within the boundaries.
I discovered that normalizing the input matrix for the activation
function could accelerate learning. Specifically, I normalized $X$ for
the Tanh function with this line:

    X = X * 2 - 1

This effectively changes the zeros in the input matrix to -1 while
leaving the ones unchanged. Since the output range of the Tanh function
is $(-1, 1)$, this normalization makes the neural network learn faster.

The second optimization was to make the neural network more predictable.
I achieved this by setting a seed for the random number generator in
NumPy:

    np.random.seed(25)

These optimizations resulted in the following output:

    With input [0 0] prediction = [0] and raw prediction = [0.09685879]
    With input [0 1] prediction = [1] and raw prediction = [0.85683077]
    With input [1 0] prediction = [1] and raw prediction = [0.85399069]
    With input [1 1] prediction = [0] and raw prediction = [0.1448133]

In my opinion, this is a very good result. The maximum number of epochs
is 500, and the learning rate remains at the allowed maximum of 0.1.

All of the text above was about solving the XOR problem, the output for
AND problem looks like this:

    With input [0 0] prediction = [0] and with raw prediction = [0.00533043]
    With input [0 1] prediction = [0] and with raw prediction = [0.05279805]
    With input [1 0] prediction = [0] and with raw prediction = [0.06984754]
    With input [1 1] prediction = [1] and with raw prediction = [0.93443729]

And for the OR problem output will be:

    With input [0 0] prediction = [0] and with raw prediction = [0.06556271]
    With input [0 1] prediction = [1] and with raw prediction = [0.93015246]
    With input [1 0] prediction = [1] and with raw prediction = [0.94720195]
    With input [1 1] prediction = [1] and with raw prediction = [0.99466957]

It can be observed that this neural network solves the AND and OR
problems more accurately than the XOR problem. This is because both AND
and OR can be solved using linear functions, whereas XOR cannot be
solved using only linear functions.

# Testing with Two Hidden Layers

The model with two hidden layers was developed after the
one-hidden-layer model. Essentially, I added a new layer with 4 neurons
as input and 4 as output, followed by the ReLU activation function. The
resulting architecture is as follows:

    layers = [
        LinearLayer(2, 4),
        Tanh(),
        LinearLayer(4, 4),
        ReLU(),
        LinearLayer(4, 1),
        Sigmoid()
    ]

This configuration produced the following output with a learning rate
(`lr`) of 0.1 and 500 epochs:

    With input [0 0] prediction = [0] and raw prediction = [0.03778528]
    With input [0 1] prediction = [1] and raw prediction = [0.97740572]
    With input [1 0] prediction = [1] and raw prediction = [0.80846681]
    With input [1 1] prediction = [0] and raw prediction = [0.03590035]

I found these results satisfactory, so no further optimizations were
made for the two-hidden-layer model. Same neural network for the OR
problem:

    With input [0 0] prediction = [0] and with raw prediction = [0.0454362]
    With input [0 1] prediction = [1] and with raw prediction = [0.99618935]
    With input [1 0] prediction = [1] and with raw prediction = [0.81249986]
    With input [1 1] prediction = [1] and with raw prediction = [0.97925493]

And for the 'AND' problem:

    With input [0 0] prediction = [0] and with raw prediction = [0.02307567]
    With input [0 1] prediction = [0] and with raw prediction = [0.04544449]
    With input [1 0] prediction = [0] and with raw prediction = [0.00353704]
    With input [1 1] prediction = [1] and with raw prediction = [0.91806489]

# Graphs

These two graphs were done while training the neural network with a
learning rate 0.1. Testing revealed that a learning rate of 0.1 is the
optimal choice for solving the XOR problem while remaining within the
appropriate boundaries for this task. Please note that all graphs below
are for the XOR problem:

<figure id="fig:tournament">
<p><img src="https://github.com/user-attachments/assets/7083314e-25a8-4b47-8d1b-7644b74e4ef0" alt="image" /> <span
id="fig:roullette" data-label="fig:roullette"></span></p>
<p><img src="https://github.com/user-attachments/assets/0f82cc74-e37d-46d6-804c-9360635b6682" alt="image" /> <span
id="fig:tournament" data-label="fig:tournament"></span></p>
</figure>

Here are the same graphs but for the learning rate 0.02:

<figure id="fig:tournament">
<p><img src="https://github.com/user-attachments/assets/cc2ef91e-b7b4-4977-a173-0693255637ab" alt="image" /> <span
id="fig:roullette" data-label="fig:roullette"></span></p>
<p><img src="https://github.com/user-attachments/assets/f8cca8e8-f8ca-4a69-9cc6-eefd579895d9" alt="image" /> <span
id="fig:tournament" data-label="fig:tournament"></span></p>
</figure>

Here are same graphs but for the model with 2 hidden layers:

<figure id="fig:tournament">
<p><img src="2with_momentum_001.png" alt="image" /> <span
id="fig:roullette" data-label="fig:roullette"></span></p>
<p><img src="2without_momentum_001.png" alt="image" /> <span
id="fig:tournament" data-label="fig:tournament"></span></p>
</figure>

<figure id="fig:tournament">
<p><img src="2without_momentum_0002.png" alt="image" /> <span
id="fig:roullette" data-label="fig:roullette"></span></p>
<p><img src="2with_momentum_0002.png" alt="image" /> <span
id="fig:tournament" data-label="fig:tournament"></span></p>
</figure>

# Evaluation of the Solution

The solution I implemented consistently provides correct predictions for
the XOR, OR, and AND problems. This accuracy is mainly due to using a
static seed for the `np.random` method, which ensures the weights are
initialized the same way every time the program runs. On top of that, I
made several optimizations to the model's architecture, which greatly
improved its performance. These changes make the model capable of
handling both linearly separable problems (like OR and AND) and more
complex, non-linearly separable problems (like XOR) with good accuracy
and efficiency.

# User Manual

To run the program, please install the required dependencies:
`matplotlib` and `numpy`. You can do this by running the following
commands in your terminal:

``` {.bash language="bash"}
pip install matplotlib
pip install numpy
```

Once the dependencies are installed, you can execute the program as
instructed.
