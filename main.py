import numpy as np
import matplotlib.pyplot as plt

# Setting seed so the random function will return same random sequence
# Random is used when generating init matrix of weights
np.random.seed(25)

# Linear layer for perceptron
class LinearLayer:
    def __init__(self, input_size, output_size):
        # Creating matrix of weight with random values
        self.weights = np.random.randn(input_size, output_size)
        # Creating vector of biases
        self.bias = np.zeros(output_size)
        self.input = None
        self.grad_weights = None
        self.grad_bias = None

        # For the momentum only
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias)

    def forward(self, x):
        self.input = x
        # Return the matrix of Xs multiplied by the weight with bias offset
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output):
        # Computing gradient for weights
        self.grad_weights = np.dot(self.input.T, grad_output)
        # Computing gradient for biases
        self.grad_bias = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.weights.T)

    def update_params(self, learning_rate, momentum):
        if momentum != 0.0:
            # Case if there is any momentum
            self.velocity_weights = momentum * self.velocity_weights - learning_rate * self.grad_weights
            self.velocity_bias = momentum * self.velocity_bias - learning_rate * self.grad_bias

            self.weights += self.velocity_weights
            self.bias += self.velocity_bias
        else:
            # Default case (without momentum)
            self.weights -= learning_rate * self.grad_weights
            self.bias -= learning_rate * self.grad_bias

# Activation function Sigmoid
class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

# Activation function Tanh
class Tanh:
    def forward(self, x):
        self.input = x
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

# Loss function MSE
class MSE:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.sum((y_pred - y_true) ** 2) / y_true.size

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size

# Activation function ReLU
class ReLU:
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.output > 0)

# Neural network
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        # Go forward through each layer in perceptron
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        # Go backward through each layer in perceptron
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

# Function to test already trained model
def test(X, model):
    X = X * 2 - 1
    y_pred = model.forward(X)
    y_pred_rounded = (y_pred > 0.5).astype(int)
    i = 0
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for x in X:
        print(f"With input {x} prediction = {y_pred_rounded[i]} and with raw prediction = {y_pred[i]}")
        i += 1

def plot_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, label="Training Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Creating list of layers in perceptron
    layers = [
        LinearLayer(2, 4),
        Tanh(),
        LinearLayer(4, 4),
        ReLU(),
        LinearLayer(4, 1),
        Sigmoid()
    ]

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_true = np.array([[0], [1], [1], [0]])
    # Normalisation of the X array, since as first activation function we are using Tanh,
    X = X * 2 - 1

    learning_rate = 0.1
    epochs = 500
    # 0.0 means that algorithm is working without momentum
    momentum = 0.9

    model = NeuralNetwork(layers)
    loss_function = MSE()
    # List for losses so we can plot graph
    losses = []
    for epoch in range(epochs):
        # Forward step for the neural network
        y_pred = model.forward(X)
        # Compute loss by MSE
        loss = loss_function.forward(y_pred, y_true)
        # Push to the list for the graph
        losses.append(loss)
        # Starting backpropagation, computing gradients from MSE
        grad_loss = loss_function.backward()
        # Go backward to compute and set parameters for the layers
        model.backward(grad_loss)
        for layer in model.layers:
            # If the layer is linear
            if hasattr(layer, 'update_params'):
                layer.update_params(learning_rate, momentum)
        print(f"Epoch number = {epoch} with loss = {loss}")
    # Create the graph for training losses
    plot_loss(losses)
    # Resetting input array
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print(f"Model is trained! Starting test with X = {X}")
    test(X, model)


if __name__ == "__main__":
    main()
