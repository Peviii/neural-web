import numpy as np
import random

class Network(object):
    def __init__(self, sizes, activations=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # Define as funções de ativação e suas derivadas para cada camada
        if activations is None:
            activations = ['sigmoid'] * (self.num_layers - 1)
        self.activations = []
        self.activation_primes = []
        for act in activations:
            if act == 'sigmoid':
                self.activations.append(self.sigmoid)
                self.activation_primes.append(self.sigmoid_prime)
            elif act == 'relu':
                self.activations.append(self.relu)
                self.activation_primes.append(self.relu_prime)
            else:
                raise ValueError(f"Ativação '{act}' não suportada.")

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def relu(self, z):
        return np.maximun(0, z)
    
    def relu_prime(self, z):
        return (z > 0).astype(float)
    
    def feedforward(self, a):
        for b, w, activation in zip(self.biases, self.weights, self.activations):
            z = np.dot(w, a) + b
            a = activation(z)
        return a
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] 

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} finalizada".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

def backpropagation(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    # Feedforward (armazenando ativações e z's)
    activation = x
    activations = [x]   # ativações de cada camada (incluindo entrada)
    zs = []             # valores z (antes da ativação) de cada camada oculta/saída

    for b, w, act in zip(self.biases, self.weights, self.activations):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = act(z)
        activations.append(activation)

    # Retropropagação (erro na saída)
    delta = self.cost_derivative(activations[-1], y) * self.activation_primes[-1](zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    # Camadas ocultas
    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = self.activation_primes[-l](z)          # derivada da ativação da camada
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

    return (nabla_b, nabla_w)

net = Network([784, 30, 30, 10], activations=['relu', 'relu', 'sigmoid'])
print(net)

import math

def softmax(logits):
    """Compute softmax for 1D or 2D input."""
    # Handle 2D list (matrix)
    if isinstance(logits, list) and logits and isinstance(logits[0], list):
        return [softmax_1d(row) for row in logits]
    # Handle 1D list
    else:
        return softmax_1d(logits)

def softmax_1d(logits):
    """Softmax for a 1D list."""
    # Find max for numerical stability
    max_z = max(logits)
    # Compute exp(logits - max_z) and sum
    exp_vals = [math.exp(z - max_z) for z in logits]
    sum_exp = sum(exp_vals)
    # Compute probabilities
    return [e / sum_exp for e in exp_vals]

def softmax_backward(d_out, logits):
    """Backward pass of softmax for 1D or 2D input."""
    # Handle 2D case
    if logits and isinstance(logits[0], list):
        return [softmax_backward_1d(d_out[i], logits[i]) for i in range(len(logits))]
    # Handle 1D case
    else:
        return softmax_backward_1d(d_out, logits)

def softmax_backward_1d(d_out, logits):
    """Backward pass for 1D softmax."""
    probs = softmax_1d(logits)
    d_input = [0.0] * len(logits)   # equivalent to vec(size, 0.0)
    n = len(probs)
    for i in range(n):
        for j in range(n):
            deriv = probs[i] * ((1.0 if i == j else 0.0) - probs[j])
            d_input[j] += d_out[i] * deriv
    return d_input

def argmax(arr):
    """Return index of maximum value in a 1D list."""
    best_i = 0
    best_v = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > best_v:
            best_v = arr[i]
            best_i = i
    return best_i

def vec(size, init=0.0):
    """Create a 1D list of given size filled with init."""
    return [init] * size

def mat(rows, cols, init=0.0):
    """Create a 2D list (rows x cols) filled with init."""
    return [vec(cols, init) for _ in range(rows)]
