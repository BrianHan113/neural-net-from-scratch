from keras.datasets import mnist
from keras.utils import to_categorical
from neural_network import NeuralNetwork

import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize pixel value from uint8 to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the images from (28, 28) to (784,1) column vector
x_train = x_train.reshape(-1, 784,1)
x_test = x_test.reshape(-1, 784,1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Reshape labels from 1D array to (10, 1) column vector
y_train = y_train.reshape(-1, 10, 1)
y_test = y_test.reshape(-1, 10, 1)

# Follow convention of keeping data and label in tuples
train_data = list(zip(x_train, y_train))
test_data = list(zip(x_test, y_test))

# # Verify dataset by displaying an image
# def display_image(image):
#     plt.imshow(image.reshape(28, 28), cmap='gray')
#     plt.axis('off')
#     plt.show()

# # Display the first image and its label from the training set
# first_image, first_label = train_data[100]
# display_image(first_image)
# print("Label:", first_label)

n = NeuralNetwork([784,15,10,10])
EPOCHS = 5
LEARNING_RATE = 0.01

n.train(train_data, EPOCHS, LEARNING_RATE)
n.evaluate(test_data)