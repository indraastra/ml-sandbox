# From https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

def display_images(inputs, outputs):
    n = len(inputs)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(inputs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(outputs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

## Autoencoder
# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
# this model maps an input to its reconstruction
autoencoder = Model(inputs=input_img, outputs=decoded)

## Encoder
# this model maps an input to its encoded representation
encoder = Model(inputs=input_img, outputs=encoded)

## Decoder
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(32,))
# retrieve the last layer of the autoencoder model
decoded_output = encoded_input
for layer in autoencoder.layers[-3:]:
    decoded_output = layer(decoded_output)
# create the decoder model
decoder = Model(inputs=encoded_input, outputs=decoded_output)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
imgs = x_test[:10]
encoded_imgs = encoder.predict(imgs)
decoded_imgs = decoder.predict(encoded_imgs)
display_images(imgs, decoded_imgs)

# Add noise and reconstruct.
noisy_test = imgs + np.random.randn(*imgs.shape) / 10
encoded_imgs = encoder.predict(noisy_test)
decoded_imgs = decoder.predict(encoded_imgs)
display_images(noisy_test, decoded_imgs)
