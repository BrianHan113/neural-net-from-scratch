from keras.datasets import mnist
from keras.utils import to_categorical
from neural_network import Network

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize pixel value from uint8 to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the images from (28, 28) to (784,1) column vector
x_train = x_train.reshape(-1, 784,1)
x_test = x_test.reshape(-1, 784,1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# print(x_train[0])
# print(y_train[0])

n = Network([784,15,10,10])
# print(n.biases[0])

print(n.forwardProp(x_train[0], y_train[0]))