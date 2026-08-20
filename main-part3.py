# Import as a module
from Perceptron_part3 import Perceptron
import matplotlib.pyplot as plt
import numpy as np

# Import MNIST
target_digit = 7

# Import data, put in suitable format
# Loading the training data
train_data = np.loadtxt("mnist_train.csv", delimiter=",")
# Replacing first letter with a bias value
train_input = [ np.append([1],d[1:]) for d in train_data ]
# Separating the labels from the image
train_label = [ int(d[0]==target_digit) for d in train_data ]

# Loading the testing data
test_data = np.loadtxt("mnist_test.csv", delimiter=",")
#test_data = np.loadtxt("mnist_test_small.csv", delimiter=",")
# Separating the labels from the image
test_input = [ np.append([1],d[1:]) for d in test_data ]
test_label = [ int(d[0]==target_digit) for d in test_data ]


# Create perceptron, 28 by 28 + a bias
p = Perceptron(28*28+1)
p.print_details()

# test untrained
p.test(test_input, test_label)

fig = plt.figure(figsize=(2,2))
data = p.weights[1:].reshape(28,28)
plt.imshow(data, cmap="inferno_r")
plt.show()

# Training the perceptron
p.train_batch(train_input, train_label)
#p.train(train_input, train_label)

# test trained
p.test(test_input, test_label)

# Save and load weights
np.save('weights.npy', data) 
#p.weights = np.load('weights.npy')

fig = plt.figure(figsize=(2,2))
data = p.weights[1:].reshape(28,28)
plt.imshow(data, cmap="inferno_r")
plt.show()



#Plot the input
# fig = plt.figure(figsize=(4,4))
# data = p.weights[1:].reshape(28,28)
# vis = train_input[0][1:].reshape(28,28)
# plt.imshow(vis)
# plt.show()

