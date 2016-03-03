import numpy as np
from scipy import sparse

from sigmoid import sigmoid

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                     num_labels, X, y, regularization):
    # Reshape weights from flattened param vectors.
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
            hidden_layer_size, (input_layer_size + 1));

    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
            num_labels, (hidden_layer_size + 1));

    # Number of datapoints.
    m = X.shape[0]

    # Feedforward computation.
    bias = np.ones((m, 1))
    a1 = np.c_[bias, X];
    z2 = a1.dot(Theta1.T);
    a2 = np.c_[bias, sigmoid(z2)];
    a3 = sigmoid(a2.dot(Theta2.T));

    # Explode the scalar labels into one-hot vector labels.
    # eg. label '3' => [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    I = np.arange(0, max(y.shape))
    J = y.ravel()
    V = np.ones(max(y.shape))
    y_exploded = np.zeros((m, num_labels));
    y_exploded[I, J] = V

    J = np.sum(np.sum(-y_exploded      * np.log(a3) -
                      (1 - y_exploded) * np.log(1 - a3))) / m;

    return J
