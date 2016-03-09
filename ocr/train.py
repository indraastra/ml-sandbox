import random

import click
import numpy as np
import scipy.io as sio
import scipy.optimize as sopt

from ex4.display_data import display_data
from ex4.nn_cost_function import nn_cost_function
from ex4.rand_initialize_weights import rand_initialize_weights
from ex4.utils import reshape_params, flatten_params, load_training_data
from ex4.predict import predict


@click.command()
@click.argument('input', type=click.Path())
@click.argument('output', type=click.Path())
@click.option('--image_pixels', default=20)
@click.option('--hidden_layer_size', default=25)
@click.option('--num_labels', default=10)
def train(input, output, image_pixels, hidden_layer_size, num_labels):
    input_layer_size  = image_pixels * image_pixels  # NxN input images

    ## === Load and visualize the training data. ===
    X, y = load_training_data(input)
    m = X.shape[0]

    # Randomly select 100 data points to display.
    print('Visualizing 100 random training data samples ...\n')
    sel = random.sample(range(m), 100)
    display_data(X[sel, :], order='C');

    ## === Initialize weights of NN randomly. ===
    print('Initializing Neural Network Parameters ...\n')

    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size);
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels);

    # Unroll parameters
    initial_nn_params = flatten_params(initial_Theta1, initial_Theta2)

    ## === Train NN. ===
    opts = {
        'maxiter': 50,
        'disp': True
    }
    progress = click.progressbar(length=opts['maxiter'])

    #  You should also try different values of lambda
    regularization = 1

    # Create "short hand" for the cost function to be minimized
    cost_function = lambda p: \
        nn_cost_function(p, input_layer_size, hidden_layer_size,
                         num_labels, X, y, regularization);
    callback = lambda x: progress.update(1)

    # Now, costFunction is a function that takes in only one argument (the
    # neural network parameters)
    print('Training Neural Network...')
    res = sopt.minimize(cost_function, initial_nn_params, callback=callback,
                        jac=True, options=opts, method='CG');
    print(res.message)
    nn_params = res.x

    # Obtain Theta1 and Theta2 back from nn_params
    Theta1, Theta2 = reshape_params(nn_params, input_layer_size,
                                    hidden_layer_size, num_labels)

    ## === Visualize NN weights. ===
    print('Visualizing Neural Network...\n')
    display_data(Theta1[:, 1:]);

    ## === Predict labels for training data ===
    pred = predict(Theta1, Theta2, X);
    accuracy = np.mean(pred == y) * 100
    print('Training Set Accuracy: {}\n'.format(accuracy));

    ## == Save weights! ==
    print('Saving out Neural Network weights.\n')
    sio.savemat(output, {'Theta1': Theta1, 'Theta2': Theta2})


if ( __name__ == '__main__' ):
    train()
