# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# 4 numpy's array
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential() # linear stack of layers
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax')) # digit probability
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255 # we want to work with float in [0,1]

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)