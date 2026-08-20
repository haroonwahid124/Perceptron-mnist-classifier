import numpy as np


class Perceptron(object):

    #==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    #==========================================#
    def __init__(self, no_inputs, max_iterations=20, learning_rate=0.1):
        self.no_inputs = no_inputs
        #self.weights = np.ones(no_inputs) / no_inputs
        self.weights = (2 * np.random.random(no_inputs) - 1) / float(255) # better to be random
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    #=======================================#
    # Prints the details of the perceptron. #
    #=======================================#
    def print_details(self):
        print("No. inputs:\t" + str(self.no_inputs))
        print("Max iterations:\t" + str(self.max_iterations))
        print("Learning rate:\t" + str(self.learning_rate))

    #=========================================#
    # Performs feed-forward prediction on one #
    # set of inputs.                          #
    #=========================================#
    def predict(self, inputs):
        # Quicker way, with dot product
        # activation = np.dot(inputs, self.weights)
        #print("act = ", activation)

        # For each weight, multiply by weight and sum
        activation_full = 0
        for i in range(0,len(inputs)):
            # Multiply
            val = inputs[i] * self.weights[i]
            # Sum
            activation_full = activation_full + val
        #print("new act ",activation_full)

        # After activation, calculate theta
        if activation_full>0:
            theta = 1
        else:
            theta = 0

        return theta

    #======================================#
    # Trains the perceptron using labelled #
    # training data.                       #
    #======================================#
    def train(self, training_data, labels):
        assert len(training_data) == len(labels)

        # Loop for number of iterations
        for its in range(self.max_iterations):
            print("epoch ", its)

            for data, label in zip(training_data, labels):
                #w_bar = w_bar + r(t-o)x_bar
                # get prediction
                prediction = self.predict(data)
                # Get error
                error = (label - prediction)

                # Error * learning rate from lecture notes
                self.weights = self.weights +self.learning_rate*error*data
            #print(self.weights)
        return

    #=========================================#
    # Tests the prediction on each element of #
    # the testing data.
    #=========================================#
    def test(self, testing_data, labels):
        assert len(testing_data) == len(labels)

        # Accuracy Count
        correct = 0
        # for each testing data
        for data, label in zip(testing_data, labels):
            #  Call prediction
            est = self.predict(data)
            print("actual ",label, "est ",est)
            if label == est:
                correct = correct+1

        # calculate accuracy, very simply
        accuracy = (correct/len(labels)) * 100
        print("Accuracy:\t"+str(accuracy))



