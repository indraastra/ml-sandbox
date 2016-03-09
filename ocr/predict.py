import click
import numpy as np
import scipy.io as sio
import scipy.optimize as sopt

from ex4.predict import predict as ex4predict
from ex4.utils import reshape_params, flatten_params, load_training_data


@click.command()
@click.argument('test_data', type=click.Path())
@click.argument('weights', type=click.Path())
@click.option('--image_pixels', default=20)
@click.option('--hidden_layer_size', default=25)
@click.option('--num_labels', default=10)
def predict(test_data, weights, image_pixels, hidden_layer_size, num_labels):
    ## === Load and visualize the test data. ===
    X, y = load_training_data(test_data)
    m = X.shape[0]

    ## === Load NN weights. ===
    print('Loading saved Neural Network parameters ...')

    # Load the weights into variables Theta1 and Theta2
    weights = sio.loadmat(weights)
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']

    # Unroll parameters 
    nn_params = flatten_params(Theta1, Theta2)

    ## === Predict labels for test data ===
    pred = ex4predict(Theta1, Theta2, X)
    accuracy = np.mean(pred == y) * 100
    print('Test Set Accuracy:', accuracy)


if ( __name__ == '__main__' ):
    predict()
