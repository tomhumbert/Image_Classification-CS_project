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
import matplotlib.pyplot as plt

## Retrieving the data from the dataloading script.

train_images, train_labels, test_images, test_labels = lpd.get_full_ds("..\\cifar-10-python\\cifar-10-batches-py")


## Decision about the amount of categories to use. (The performance of the script to collect 5 categories is sub-optimal.)

ans = int(input("How many categories? (2, 5 or 10)"))

if ans == 2:
    train_lbls, train_imgs = lpd.get_categories(train_images, train_labels, ['airplane', 'dog'])
    test_lbls, test_imgs = lpd.get_categories(test_images, test_labels, ['airplane', 'dog'])
    
if ans == 5:
    train_lbls, train_imgs = lpd.get_categories(train_images, train_labels, ['airplane', 'dog', 'bird', 'cat', 'ship'])
    test_lbls, test_imgs = lpd.get_categories(test_images, test_labels, ['airplane', 'dog', 'bird', 'cat', 'ship'])
    
else:
    train_lbls, train_imgs = train_labels, train_images
    test_lbls, test_imgs = test_labels, test_images

train_lbls = to_categorical(train_lbls, num_classes=10)
test_lbls = to_categorical(test_lbls, num_classes=10)


## Model creation.

network = models.Sequential() 
network.add(layers.Dense(1000, activation='relu', input_shape=(32 * 32 * 3,)))
network.add(layers.Dense(500, activation='elu'))
#network.add(layers.Dropout(0.3))
network.add(layers.Dense(100, activation='elu'))
network.add(layers.Dense(10, activation='softmax')) 

opt = tf.keras.optimizers.Adamax(
    learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax")

network.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])


## Training and subsequent testing

history = network.fit(train_imgs, train_lbls, epochs=10, batch_size=32)
test_loss, test_acc = network.evaluate(test_imgs, test_lbls)


## Plotting the results from testing, namely the improvement of accuracy over time

plt.plot(history.history['accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

## Exporting a vizualization of the model

dot_img_file = './network.png'
tf.keras.utils.plot_model(network, to_file=dot_img_file, show_shapes=True)
