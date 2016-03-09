import numpy as np

def reshape_params(nn_params, input_layer_size, hidden_layer_size, num_labels):
    # Reshape weights from flattened param vectors.
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
            hidden_layer_size, (input_layer_size + 1))

    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
            num_labels, (hidden_layer_size + 1))

    return Theta1, Theta2

def flatten_params(Theta1, Theta2):
    return np.concatenate((Theta1.ravel(), Theta2.ravel()))
