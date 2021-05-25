# -*- coding: utf-8 -*-
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""  Tom Humbert                                                 """
"""  CS Project - Image Classification                           """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import load_process_data as lpd

train_images, train_labels, test_images, test_labels = lpd.get_full_ds("..\\cifar-10-python\\cifar-10-batches-py")

train_lbls, train_imgs = lpd.get_categories(train_images, train_labels, ['airplane', 'dog'])
test_lbls, test_imgs = lpd.get_categories(test_images, test_labels, ['airplane', 'dog'])

train_lbls = to_categorical(train_lbls, num_classes=10)
test_lbls = to_categorical(test_lbls, num_classes=10)

network = models.Sequential() 
network.add(layers.Dense(512, activation='relu', input_shape=(32 * 32 * 3,)))
network.add(layers.Dense(10, activation='softmax')) 
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

network.fit(train_imgs, train_lbls, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_imgs, test_lbls)

print(test_acc)