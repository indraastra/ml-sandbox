import os
import random
import string

import numpy as np
from PIL import ImageFont, ImageDraw
from PIL import Image

FONT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts')
FONT_BLACKLIST = set([
    'Corben-Bold.ttf',
    'FiraMono-Regular.ttf',
    'FiraMono-Medium.ttf',
    'FiraMono-Bold.ttf',
])

def get_fonts():
    return [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR)
            if f.endswith(".ttf") and f not in FONT_BLACKLIST]

def load_font(path, size=64):
    return ImageFont.truetype(path, size)

def glyph_to_image(glyph, font, size=64, mode='L'):
    im = Image.new(mode, (size, size), 'black')
    draw = ImageDraw.Draw(im)
    # Vertically and horizontally center glyph in canvas, accounting for offset.
    w, h = font.getsize(glyph)
    x, y = font.getoffset(glyph)
    draw.text(((size-w-x)/2, (size-h-y)/2), glyph, font=font, fill='white')
    return im

def image_to_numpy(image):
    return np.array(image.getdata(), np.float64) / 255

def numpy_to_image(array):
    pass


if (__name__ == '__main__'):
    fonts = get_fonts()
    print(len(fonts), "available fonts to choose from.")
    font_path = random.choice(fonts)
    print("Randomly selecting", font_path)
    font = load_font(font_path, 64)
    glyph = random.choice(string.digits)
    print("Showing random glyph in this font:", glyph)
    im = glyph_to_image(glyph, font, 72)
    im.show()
