from neuron.neuron_single_pass import neuron_single_pass
from neuron.artificial_neuron import Neuron
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# Add the parent directory to the Python path so we can import the neuron
# module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def generate_temp_data():
    '''
    Generate synthetic temperature data for training.
    fixed values for reproducibility in accordance with the article results.
    '''
    input_list = [i for i in range(1, 25)]  # Values from 1 to 24
    ground_truth_list = [26, 23, 25, 24.6, 26, 24.5, 25, 23.7, 22, 27, 23, 25.6,
                         26, 25, 26, 24.7, 24, 25, 27, 24.6, 24, 27, 25, 23.7]
    return input_list, ground_truth_list


def plot_loss(losses):
    '''
    Plot the loss over epochs.
    '''
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title('Neuron Error (Loss) over Epochs')
    plt.xlabel("Training Epochs")
    plt.ylabel("Error (Loss)")
    plt.show()


def poly_fit(input_data, ground_truth_data):
    '''
    Fit a polynomial to the data for comparison using the least squares method.
    This will help visualize the optimal solution.
    '''
    # Fit a line of best fit (1st degree polynomial)
    coefficients = np.polyfit(input_data, ground_truth_data, 1)
    polynomial = np.poly1d(coefficients)

    # Generate y-values for the line of best fit
    line_of_best_fit = polynomial(input_data)
    return line_of_best_fit


def plot_neuron_results_and_poly_fit(neuron, input_data, ground_truth_data):
    '''
    Plot the neuron's output against the ground truth.
    '''
    neuron_outputs = []
    for input_value in input_data:
        neuron.input = input_value
        neuron.forward_pass()
        neuron_outputs.append(neuron.output)

    poly_fit_line = poly_fit(input_data, ground_truth_data)

    plt.scatter(
        input_data,
        ground_truth_data,
        color='blue',
        label='Hourly Temperature Readings')
    plt.plot(
        input_data,
        neuron_outputs,
        color='Green',
        linestyle='-',
        label=f'Neuron Approximation: Y = {neuron.weight:.4f} * X + {neuron.bias:.4f} ')
    # Plot the line of best fit
    plt.plot(input_data, poly_fit_line, color='red', linestyle='-',
             label='Optimal solution: Y = 0.0406 * X + 24.301 ')
    plt.title("Temperatures throughout a day")
    plt.xlabel("Hours (X)")
    plt.ylabel("Temperatures °C (Y)")
    plt.legend()
    plt.ylim(20, 30)
    plt.show()


if __name__ == "__main__":
    # Hyperparameters for the neuron
    NEURON_STARTING_WEIGHT = 1.0
    NEURON_STARTING_BIAS = 25.0
    NEURON_LEARNING_RATE = 0.0001

    # Hyperparameters for the training
    # Number of times the neuron will be trained on the data
    TRAINING_EPOCHS = 10

    # Generate synthetic temperature data
    input_data, ground_truth_data = generate_temp_data()

    # Initialize the neuron
    neuron = Neuron(starting_weight=NEURON_STARTING_WEIGHT,
                    starting_bias=NEURON_STARTING_BIAS,
                    learning_rate=NEURON_LEARNING_RATE)

    # Train the neuron with the generated data
    losses = []
    for epoch in range(TRAINING_EPOCHS):
        epoch_loss = 0
        for input_value, ground_truth_value in zip(
                input_data, ground_truth_data):
            loss = neuron_single_pass(neuron, input_value, ground_truth_value)
            epoch_loss += loss
        losses.append(epoch_loss)

    # Plot the results
    plot_loss(losses)
    plot_neuron_results_and_poly_fit(neuron, input_data, ground_truth_data)
    print(
        f"Final Neuron Parameters: Weight = {neuron.weight:.4f}, Bias = {neuron.bias:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")
