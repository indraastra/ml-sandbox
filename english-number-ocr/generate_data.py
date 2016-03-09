#!/usr/bin/env python

import itertools

import click
import numpy as np
import scipy.io as sio

from en_utils import get_fonts, load_font, glyph_to_image, image_to_numpy


@click.command()
@click.argument('output')
def generate_data(output):
    fonts = get_fonts()[:10]
    glyphs = '0123456789'
    imscale = 20

    # Allocate space for training data.
    H = len(fonts) * len(glyphs)  # Number of training instances.
    W = imscale * imscale         # Pixels in image.
    X = np.zeros((H, W), dtype=np.float64)
    y = np.zeros((H, 1), dtype=np.uint8)

    # Generate all numbers in all fonts.
    print("Generating font-based training data...")
    with click.progressbar(fonts) as pb:
        for i, font_path in enumerate(pb):
            font = load_font(font_path, imscale - 4)
            for j, glyph in enumerate(glyphs):
                try:
                    im = glyph_to_image(glyph, font, imscale)
                    im_np = image_to_numpy(im)
                    idx = i * len(glyphs) + j
                    X[idx, :] = im_np
                    y[idx] = j
                except OSError:
                    print("Error with font:", font_path)
                    break

    sio.savemat(output, {'X': X, 'y': y})
    click.echo('Successfully created training data!')


if (__name__ == '__main__'):
    generate_data()
