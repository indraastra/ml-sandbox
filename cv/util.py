import numpy as np

def pil_image_to_np_array(img):
  return np.array(img.getdata(), np.uint8).reshape(img.height, img.width, 3)
