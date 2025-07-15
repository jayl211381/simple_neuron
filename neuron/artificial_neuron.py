import numpy as np


class Neuron:
    def __init__(self, starting_weight=1.0, starting_bias=1.0,
                 learning_rate=0.01, input=0.0, ground_truth=0.0):
        # Initialize the neuron parameters nessessary for training
        self.input = input
        self.ground_truth = ground_truth
        self.weight = starting_weight
        self.bias = starting_bias
        self.learning_rate = learning_rate
        # Intermediate variables for forward and backward pass
        self.weight_grad = 0.0
        self.bias_grad = 0.0
        self.logit = 0.0
        self.output = 0.0

    @staticmethod
    def relu(x):
        '''
        Rectified Linear Unit activation function.
        Returns x if x > 0, otherwise returns 0.
        '''
        return np.maximum(0, x)

    @staticmethod
    def relu_prime(x):
        '''
        Derivative of the Rectified Linear Unit activation function.
        Returns 1 for x > 0, otherwise returns 0.
        '''
        return np.where(x > 0, 1, 0)

    def forward_pass(self):
        '''
        Forward pass through the neuron.
        Computes the logit and the output using the ReLU activation function.
        '''
        self.logit = self.weight * self.input + self.bias
        self.output = self.relu(self.logit)

    def mse_error(self):
        '''
        Mean Squared Error loss function.
        Computes the error between the output and the ground truth.
        '''
        return 0.5 * (self.ground_truth - self.output)**2

    def back_prop(self):
        '''
        Backward pass through the neuron.
        Computes the gradients of the loss with respect to the neuron's parameters.
        '''
        # Calculate the derivative of the loss with respect to the output
        loss_gradient = -(self.ground_truth - self.output)

        # Apply chain rule with ReLU derivative
        neuron_gradient = loss_gradient * self.relu_prime(self.logit)

        # Calculate gradients
        self.bias_grad = neuron_gradient
        self.weight_grad = self.input * neuron_gradient

    def optimizer(self):
        '''
        Optimizer function to update the neuron's parameters.
        '''
        self.weight = self.weight - self.weight_grad * self.learning_rate
        self.bias = self.bias - self.bias_grad * self.learning_rate
