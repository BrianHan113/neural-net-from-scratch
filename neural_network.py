import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Each neuron has its own bias
        self.biases = [np.random.randn(neurons, 1) for neurons in sizes[1:]]
        # Each neuron has n-number of weights, where n is num of neurons of previous layer
        # sizes <- [2,3,3,4] => zip([2,3,3], [3,3,4]) => ([2,3], [3,3], [3,4])
        self.weights = [np.random.randn(current_layer, prev_layer) 
                        for prev_layer, current_layer in zip(sizes[:-1], sizes[1:])]
        # weights[0] == weights of first non-input layer, 
        # weights[1] == weights of 2nd non-input layer, 
        # etc, same for biases

        self.zs = [] # Store calculated Z values for each layer during forwardProp
    
    def forwardProp(self, input, y):
        # Clear previous Z values
        self.zs = []
        A = input
        layer_num = 1 # Assuming input layer is layer 0

        for b, w in zip(self.biases, self.weights):

            z = np.dot(w, A) + b
            self.zs.append(z)
            
            # Apply softmax to the output layer and return the cost
            if (layer_num == len(self.sizes) - 1):
                A = softmax(z)
                print("Softmax layer {}".format(layer_num))
                print(A)
                return mse(A, y)
            # Otherwise, apply ReLU
            else:
                A = relu(z)
                print("Relu layer {}".format(layer_num))
                print(A)
                layer_num += 1
                continue

    def backProp(self):
        print('ruh roh')

    def train(self, training_data, epochs, learning_rate):
        return


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    # print(exp_z / np.sum(exp_z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)