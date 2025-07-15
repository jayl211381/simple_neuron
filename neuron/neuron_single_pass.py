def neuron_single_pass(neuron, input, ground_truth):
    '''
    Train the neuron using a single forward and backward pass.
    '''

    # Set input and ground truth
    neuron.input = input
    neuron.ground_truth = ground_truth

    # Forward pass
    neuron.forward_pass()

    # Compute loss
    loss = neuron.mse_error()

    # Backward pass
    neuron.back_prop()

    # Update weights and bias
    neuron.optimizer()

    return loss
