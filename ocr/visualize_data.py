#!/usr/bin/env python

import random

import click
import numpy as np

from ex4.display_data import display_data
from ex4.utils import load_training_data


## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

@click.command()
@click.argument('input')
def visualize_data(input):
    # Load Training Data
    print('> Loading and Visualizing Data ...\n')

    X, _ = load_training_data(input)
    m = X.shape[0]

    # Randomly select 100 data points to display
    sel = random.sample(range(m), 100)

    display_data(X[sel, :], order='C');


if (__name__ == '__main__'):
    visualize_data()
