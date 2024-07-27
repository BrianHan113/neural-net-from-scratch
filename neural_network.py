import numpy as np

class NeuralNetwork(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Each neuron has its own bias
        self.biases = [np.random.randn(neurons, 1) for neurons in sizes[1:]]
        # Each neuron has n-number of weights, where n is num of neurons of previous layer
        # e.g. sizes <- [2,3,3,4] => zip([2,3,3], [3,3,4]) => ([2,3], [3,3], [3,4])
        self.weights = [np.random.randn(current_layer_neurons, prev_layer_neurons) 
                        for prev_layer_neurons, current_layer_neurons in zip(sizes[:-1], sizes[1:])]
        # weights[0] == weights of first non-input layer, 
        # weights[1] == weights of 2nd non-input layer, 
        # etc, same for biases
    
    def forwardProp(self, input, y):
        zs = [] # Store Z values
        As = [] # Store A values
        A = input
        layer_num = 1 # Assuming input layer is layer 0

        for b, w in zip(self.biases, self.weights):

            z = np.dot(w, A) + b
            zs.append(z)
            
            # Apply softmax to the output layer and return the cost
            if (layer_num == len(self.sizes) - 1):
                A = softmax(z)
                As.append(A)
                # print("Softmax layer {}".format(layer_num))
                # print(A)
                cost = mse(A, y)
                return (cost, zs, As)
            # Otherwise, apply ReLU
            else:
                A = relu(z)
                As.append(A)
                # print("Relu layer {}".format(layer_num))
                # print(A)
                layer_num += 1
                continue

    def backProp(self, zs, As):
        # TODO: implement
        partial_w = -1
        partial_b = -1
        return (partial_w, partial_b)
    
    def updateParams(self, partial_w, partial_b, learning_rate):
        # TODO: implement
        return

    def train(self, train_data, epochs, learning_rate):
        for epoch in range(1, epochs+1):
            print("Epoch: " + str(epoch))
            np.random.shuffle(train_data)
            
            costs = [] # Store the cost of each training example per epoch

            testing = 0
            for sample in train_data:

                cost, zs, As = self.forwardProp(sample[0], sample[1])
                costs.append(cost)
                partial_w, partial_b = self.backProp(zs, As)
                self.updateParams(partial_w, partial_b, learning_rate)

                testing += 1
                if (testing == 2):
                    break
            
            print(costs)
            
        return


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)