import argparse
import math
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


parser = argparse.ArgumentParser(description='Average an image.')
parser.add_argument('image', metavar='I', type=str,
                    help='the image to quantize')
parser.add_argument('neighborhood', metavar='N', type=int,
                    help='the LxL neighborhood of pixels to average over')


if __name__ == '__main__':
  args = parser.parse_args()

  N = args.neighborhood
  print('Average {} with neighborhood of {} pixels'.format(args.image, N))

  image = mpimg.imread(args.image)

  channels = []
  for i in range(3):
    channel = image[:, :, i]
    channels.append(convolve(channel, np.ones((N, N)) / (N ** 2), mode='constant'))
  avg_image = np.dstack(channels)

  plt.subplot(211)
  plt.axis('off')
  plt.imshow(image)

  plt.subplot(212)
  plt.axis('off')
  plt.imshow(avg_image)

  plt.show()
