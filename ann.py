import random
import numpy as np

# Network class
class Network:
    
    def __init__(self, dim):
        '''Initialize neural network with a list of how many neurons there are in each layer.'''
        self.num_layers = len(dim)
        self.biases = [np.random.randn(x) for x in dim[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(dim[:-1], dim[1:])]

    def feedforward(self, vect):
        '''Compute the output of the neural network for the given input.'''
        for b, w in zip(self.biases, self.weights):
            vect = sigmoid(np.dot(w, vect) + b)
        return vect

    def backprop(self, training_data, epochs, lr, test_data = None):
        '''Train the neural network with a given training data set, number of epochs, and learning rate.
        A testing data set can be given to evaluate the network's performance at the end of each epoch.'''
        for epoch in range(epochs):
            db = [np.zeros(b.shape) for b in self.biases]
            dw = [np.zeros(w.shape) for w in self.weights]
            for x, y in training_data:
                activation = x
                activations = [x]
                zs = []
                # Compute z and activation vectors for each layer
                for b, w in zip(self.biases, self.weights):
                    z = np.dot(w, activation) + b
                    zs.append(z)
                    activation = sigmoid(z)
                    activations.append(activation)
                # Compute the error vector for the last layer
                delta = 2*(activation - y)*sigmoid_prime(z)
                # Backpropagate the error
                for l in range(self.num_layers - 1, 0, -1):
                    dw[l - 1] += np.dot(np.matrix(delta).T, np.matrix(activations[l - 1]))
                    db[l - 1] += delta
                    if l > 1:
                        delta = np.dot(delta, self.weights[l - 1])*sigmoid_prime(zs[l - 2])
            # Update the weights and biases
            n = len(training_data)
            for b, w, ldb, ldw in zip(self.biases, self.weights, db, dw):
                b -= lr*ldb/n
                w -= lr*ldw/n
            # Show results
            if test_data:
                print("Epoch", epoch + 1, "/", epochs, "| Performance:", self.evaluate(test_data), "%")
            else:
                print("Completed epoch", epoch + 1, "/", epochs)

    def evaluate(self, test_data):
        '''Return the percentage representing how much of the test data was correctly computed.'''
        return (sum(np.rint(self.feedforward(x)) == y for x, y in test_data)/len(test_data)*100)[0]

# Utilities
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))