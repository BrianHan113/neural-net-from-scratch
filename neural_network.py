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
    
    def forwardProp(self, sample):

        input = sample[0]
        y = sample[1]

        zs = [] # Store Z values
        As = [] # Store A values
        A = input
        As.append(A)
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

    def backProp(self, zs, As, y):

        # y = y.reshape(-1, 1)

        # Initialize gradients
        partial_w = [np.zeros(w.shape) for w in self.weights]
        partial_b = [np.zeros(b.shape) for b in self.biases]

        # Output layer
        partial_z = np.dot(softmaxPrime(zs[-1]), msePrime(As[-1], y))
        partial_w[-1] = np.dot(partial_z, As[-2].transpose())
        partial_b[-1] = partial_z

        # Propogate backwards through each layer, up to the input layer
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            partial_A_partial_z = reluPrime(z)

            # Update partial C / partial z - reuse for future layers to save computation
            partial_z = np.dot(self.weights[-layer+1].transpose(), partial_z) * partial_A_partial_z
            partial_w[-layer] = np.dot(partial_z, As[-layer-1].transpose())
            partial_b[-layer] = partial_z
        return (partial_w, partial_b)
    
    def updateParams(self, partial_w, partial_b, learning_rate):
        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * partial_w[layer]
            self.biases[layer] -= learning_rate * partial_b[layer]

    def train(self, train_data, epochs, learning_rate):
        for epoch in range(1, epochs+1):
            print("Epoch: " + str(epoch))
            np.random.shuffle(train_data)
            
            costs = [] # Store the cost of each training example per epoch

            testing = 0
            for sample in train_data:

                cost, zs, As = self.forwardProp(sample)
                costs.append(cost)

                # partial_x naming refers to the derivative: partial C / partial x
                partial_w, partial_b = self.backProp(zs, As, sample[1])
                self.updateParams(partial_w, partial_b, learning_rate)

                # testing += 1
                # if (testing == 1):
                #     break
            
            print("Average cost: ", np.mean(costs))
        return
    
    def evaluate(self, test_data):
        correct_count = 0
        for sample in test_data:
            _, _, A = self.forwardProp(sample)
            y_pred = A[-1]

            if (np.argmax(y_pred) == np.argmax(sample[1])):
                correct_count += 1

        print("{:.2f}% Accuracy".format((correct_count / len(test_data)) * 100))


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def msePrime(y_pred, y_true):
    # Dont include the 1/m factor - see notes
    return 2*(y_pred - y_true)

def relu(z):
    return np.maximum(0, z)

def reluPrime(z):
    # Derivative of ReLU is a step function
    # Return 1 for z > 0, otherwise return 0
    return np.where(z > 0, 1, 0)

def softmax(z):
    # -max improves numerical stability
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def softmaxPrime(z):
    # honestly, i copy pasted this one
    s = softmax(z)
    s = s.reshape(-1, 1)
    s_j = s.T
    J = np.diagflat(s.flatten()) - np.dot(s, s_j)
    return J