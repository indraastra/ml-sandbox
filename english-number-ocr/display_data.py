import math

import numpy as np
import scipy.misc as smisc

def display_data(images):
    # Select as many images as can neatly fit in a square display.
    w = int(math.sqrt(len(images)))
    h = int(len(images) / w)
    images = images[:w*h]
    print("Displaying images in a {}x{} array.".format(h, w))

    size = int(math.sqrt(images[0].shape[0]))

    combined_image = np.zeros((size * h, size * w))

    for i in range(h):
        for j in range(w):
            image = images[i * w + j]
            start_x = i * size;
            start_y = j * size;
            combined_image[start_y:(start_y + size),
                           start_x:(start_x + size)] = image.reshape((size, size), order='F')
    smisc.imshow(combined_image)
