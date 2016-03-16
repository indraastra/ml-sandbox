import click
import numpy as np
import scipy.io as sio
import scipy.optimize as sopt

from ex4.predict import predict as ex4predict
from ex4.utils import reshape_params, flatten_params, load_training_data

def load_classifier(weights_file):
    # Load the weights into variables Theta1 and Theta2
    weights = sio.loadmat(weights_file)
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']

    ## === Predict labels for test data ===
    return lambda X: ex4predict(Theta1, Theta2, X)


@click.command()
@click.argument('test_data', type=click.Path())
@click.argument('weights', type=click.Path())
@click.option('--source', default='numpy',
              type=click.Choice(['numpy', 'matlab']))
def predict(test_data, weights, source):
    ## === Load and visualize the test data. ===
    X, y = load_training_data(test_data, source)

    ## === Load NN weights. ===
    print('Loading saved Neural Network parameters ...')
    classify = load_classifier(weights)

    ## === Predict labels for test data ===
    pred = classify(X)
    accuracy = np.mean(pred == y) * 100
    print('Test Set Accuracy:', accuracy)

    example = (X[0, :].reshape(20, 20, order='F').round().astype(np.uint8)) 
    print(example)
    pred = classify(example.reshape(1, 400, order='F'))
    print(pred, y[0])

if ( __name__ == '__main__' ):
    predict()
