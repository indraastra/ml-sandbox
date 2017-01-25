# Adapted from http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
# and https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d

import os

from lenet import LeNet
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras import backend as K
from imutils import paths
import numpy as np
import argparse

from clog import clog


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--save_model', action='store_true',
    help='(optional) whether or not model should be saved to disk')
ap.add_argument('-l', '--load_model', action='store_true',
    help='(optional) whether or not pre-trained model should be loaded')
ap.add_argument('-w', '--weights_file', type=str, help='(optional) path to weights file')
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-c', '--classify_image', type=str, help='(optional) path to image to predict')
args = vars(ap.parse_args())

# dimensions of our images.
img_width, img_height = 150, 150
train_data_dir = os.path.join(args['dataset'], 'train')
validation_data_dir = os.path.join(args['dataset'], 'validation')
test_data_dir = os.path.join(args['dataset'], 'minitest')
nb_epoch = 100
nb_train_samples = 2000
nb_validation_samples = 800

class_labels = os.listdir(train_data_dir)
class_labels.sort()
clog.info('Classes: {}'.format(class_labels))

# Initialize the optimizer and model.
clog.info('Initializing model...')
model = LeNet.build(width=img_width, height=img_height, depth=3, num_classes=2,
    weights_path=args['weights_file'] if args['load_model'] > 0 else None)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
    metrics=['accuracy'])

# Only train and evaluate the model if we *are not* loading a
# pre-existing model.
if not args['load_model']:
  # this is the augmentation configuration we will use for training
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

  # this is the augmentation configuration we will use for testing:
  # only rescaling
  validation_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

  validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

  clog.info('Training...')
  model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)


# Check to see if the model should be saved to file.
if args['save_model'] > 0:
  clog.info('Dumping weights to file...')
  model.save_weights(args['weights_file'], overwrite=True)


if args['classify_image']:
  # Evaluate single image.
  img = load_img(args['classify_image'])
  img = img.resize((img_width, img_height))
  np_img = img_to_array(img)
  np_img = np_img[np.newaxis, :, :, :]

  # classify the image
  probs = model.predict(np_img)
  prediction = np.argmax(probs)

  # show the prediction
  clog.info('For image [{}], predicted {} with probability {}'.format(
      args['classify_image'], class_labels[prediction], probs[0, prediction]))
else:
  # Evaluate on test set.
  test_dataggen = ImageDataGenerator(rescale=1./255)
  test_generator = test_dataggen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

  loss, accuracy = model.evaluate_generator(test_generator, val_samples=1024)
  print("Accuracy = {:.2f}".format(accuracy))

K.clear_session()
