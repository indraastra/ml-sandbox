import argparse
import math
import time
from PIL import Image

parser = argparse.ArgumentParser(description='Quantize an image.')
parser.add_argument('image', metavar='I', type=str,
                    help='the image to quantize')
parser.add_argument('levels', metavar='L', type=int,
                    help='the number of levels to quantize to; must be a power of 2')


if __name__ == '__main__':
  args = parser.parse_args()

  print('Quantizing {} to {} levels'.format(args.image, args.levels))

  with Image.open(args.image) as image:
    image.show(title='Original')
    time.sleep(.1)

    quant_image = image.point(lambda p: math.floor(p / args.levels) * args.levels)
    quant_image.show(title='Quantized')

  input('Press ENTER to exit...')
