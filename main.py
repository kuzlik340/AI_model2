import numpy as np
import matplotlib.pyplot as plt

class LinearLayer:
    def __init__(self, input_size, output_size):
        # Creating matrix of weight with random values
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        # Creating vector of biases
        self.bias = np.zeros(output_size)
        self.input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        self.input = x
        #print(f"X from linear layer = {x}")
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output):
        # Computing gradient for weights
        self.grad_weights = np.dot(self.input.T, grad_output)
        # Computing gradient for biases
        self.grad_bias = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input

class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)


class MSE:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

class ReLU:
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output * (self.output > 0)
        return grad_input


def main():
    # Setting seed so the random function will return same random sequence
    # Random is used when generating init matric of weights
    np.random.seed(25)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_true = np.array([[0], [1], [1], [0]])
    # Normalisation of the X array, since as first activation function we are using Tanh,
    X = X * 2 - 1
    layers = [
        LinearLayer(2, 4),
        Tanh(),
        # LinearLayer(4, 4),
        # ReLU(),
        LinearLayer(4, 1),
        Sigmoid()
    ]
    learning_rate = 0.1
    epochs = 500
    model = NeuralNetwork(layers)
    loss_function = MSE()
    y_pred = None
    for epoch in range(epochs):
        y_pred = model.forward(X)
        #print(X)
        loss = loss_function.forward(y_pred, y_true)
        grad_loss = loss_function.backward()
        model.backward(grad_loss)
        for layer in model.layers:
            if hasattr(layer, 'update_params'):
                layer.update_params(learning_rate)
        print(f"Epoch number = {epoch} with loss = {loss}")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_pred_rounded = (y_pred > 0.5).astype(int)
    i = 0
    for x in X:
        print(f"With input {x} prediction = {y_pred_rounded[i]} and with raw prediction = {y_pred[i]}")
        i += 1


if __name__ == "__main__":
    main()
