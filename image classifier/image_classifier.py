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
from datetime import datetime


## Decision about the amount of categories to use. (The performance of the script to collect 5 categories is sub-optimal.)
def set_ds(number_categories=10, data_location="..\\cifar-10-python\\cifar-10-batches-py"):
    """
    This function retrieves the given number of categories.

    Parameters
    ----------
    number_categories : Integer, optional
        Can be 2, 5 or 10. Any other value results in choosing 10. The default is 10.
    data_location : String, optional
        The location of the dataset folder. The default value is the relative path to the dataset within the project folder.

    Returns
    -------
    train_imgs : Numpy array
        An array of training images.
    train_lbls : Categorical boolean matrix
        A boolean matrix with the category data of the according image.
    test_imgs : Numpy array
        An array of testing images.
    test_lbls : Categorical boolean matrix
        A boolean matrix with the category data of the according image.

    """
    train_images, train_labels, test_images, test_labels = lpd.get_full_ds(data_location)
    
    if number_categories == 2:
        train_lbls, train_imgs = lpd.get_categories(train_images, train_labels, ['airplane', 'dog'])
        test_lbls, test_imgs = lpd.get_categories(test_images, test_labels, ['airplane', 'dog'])
        
    elif number_categories == 5:
        train_lbls, train_imgs = lpd.get_categories(train_images, train_labels, ['airplane', 'dog', 'bird', 'cat', 'ship'])
        test_lbls, test_imgs = lpd.get_categories(test_images, test_labels, ['airplane', 'dog', 'bird', 'cat', 'ship'])
        
    else:
        train_lbls, train_imgs = train_labels, train_images
        test_lbls, test_imgs = test_labels, test_images
    
    train_lbls = to_categorical(train_lbls, num_classes=10)
    test_lbls = to_categorical(test_lbls, num_classes=10)
    
    print(train_imgs.shape)
    
    return train_imgs, train_lbls, test_imgs, test_lbls


## Model creation.

def create_model(optimizer, activation, has_drop):
    """
    This function creates the neural network model. The parameters are only applied on the hidden layers.
    
    Parameters
    ----------
    optimizer : TF optimizer object
        Choice of optimizer.
    activation : String
        Choice of activation function.
    has_drop : Bool
        Determines if dropout layers should be added.
    has_noise : Bool
        Determines if noise layers should be added.

    Returns
    -------
    A Keras model object.

    """
    network = models.Sequential() 
    
    # input
    network.add(layers.Dense(1000, activation='relu', input_shape=(32 * 32 * 3,)))
    
    # hidden

    network.add(layers.Dense(500, activation=activation))
    if has_drop:
        network.add(layers.Dropout(0.3))
        
    
    network.add(layers.Dense(100, activation=activation))
    if has_drop:
        network.add(layers.Dropout(0.3))
    
    # output
    network.add(layers.Dense(10, activation='softmax')) 
    
    network.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return network


## Training and subsequent testing.

def train(model, train_imgs, train_lbls, epochs, batch_size):
    """
    The function that launches the training of the model.

    Parameters
    ----------
    model : Model object
        A Keras Model object.
    train_imgs : numpy array
        A numpy array containing the color channel values of an image.
    train_lbls : boolean matrix
        A matrix containing the categories according to the training images.
    epochs : Integer
        Number of epochs the model will be trained for.
    batch_size : Integer
        The batch size of the model training.

    Returns
    -------
    model : Model object
        A Keras Model object.
    history : history object
        A Keras Model training history object.
    t_diff : float
        The time past during training.

    """
    
    t0 = datetime.now()
    history = model.fit(train_imgs, train_lbls, epochs=epochs, batch_size=batch_size)
    t1 = datetime.now() 
    t_diff = t1 - t0
    
    return model, history, t_diff
    

def test(model, test_imgs, test_lbls):
    """
    The function that launches the validation process of the model.

    Parameters
    ----------
    model : Model object
        A Keras Model object.
    test_imgs : numpy array
        A numpy array containing the color channel values of images.
    test_lbls : boolean matrix
        A matrix containing the categories according to the test images.

    Returns
    -------
    test_loss : float
        Value of the loss function during validation.
    test_acc : float
        Accuracy value recorded during validation.

    """
    test_loss, test_acc = model.evaluate(test_imgs, test_lbls)
    
    return test_loss, test_acc


## Plotting the results from testing, namely the improvement of accuracy over time.

def plot_performance(history):
    # from https://www.codesofinterest.com/2017/03/graph-model-training-history-keras.html
    plt.figure(1)  
       
    # summarize history for accuracy  
       
    plt.subplot(211)  
    plt.plot(history.history['accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
       
     # summarize history for loss  
       
    plt.subplot(212)  
    plt.plot(history.history['loss'])   
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()  
    
    ## Exporting a vizualization of the model.

def plot_model(model):
    dot_img_file = '../network.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(num, opt, act, has_drop, epochs, batch_size, do_plot_accuracy, do_plot_model):
    
    print(f"Start collecting data for {num} categories\n")
    train_imgs, train_lbls, test_imgs, test_lbls = set_ds(number_categories=num)
    print("Collection of data done.\nProceding with model generation..\n")
    
    model = create_model(opt, act, has_drop)
    print(f"Created model with {opt} optimizer, {act} activation function for hidden layers and \ndropout set to {has_drop}\n")
    
    if do_plot_model:
        plot_model(model)
        print("The model has been visualized. The file is saved in the project folder.\n")
    
    print(f"Starting model training with {epochs} epochs and a batch size of {batch_size}\n")
    model, history, t_diff = train(model, train_imgs, train_lbls, epochs, batch_size)
    print(f"The model finished training. It took {t_diff}.\n")
    
    print("Starting validation with testing images.\n")
    test_loss, test_acc = test(model, test_imgs, test_lbls)
    print(f"Testing finished with an accuracy of {test_acc}.\n")
    
    if do_plot_accuracy:
        plot_performance(history)
        print("A plot for the accuracy improvement over time has been created\n")
    

if __name__ == "__main__":
    
    print("Neural Network creation and testing routine. If you wish to not change the standard parameters, just click enter at each question.")
    
    try:
        number_of_categories = int(input("Do you wish to test with 2, 5 or 10 categories?\n Answer: "))
        if number_of_categories not in [2, 5, 10]:
            number_of_categories = 10
    except:
        number_of_categories = 10
        
        
    act = input("Which activation function for hidden layers? (relu, elu,..)\n Answer: ")
    if act not in ['elu', 'relu']:
        act = 'elu'
    
    opt = input("Which optimizer? (adam, adamax, ..)\n Answer: ")
    if opt not in ['adam', 'adamax', 'rmsprop']:
        opt = tf.keras.optimizers.Adamax(
            learning_rate=0.0005, 
            beta_1=0.9, beta_2=0.999, 
            epsilon=1e-07
            )
    
    q1 = input("Do you want dropout layers? (y,n)\n Answer: ")
    if q1 == 'y':
        has_drop = True
    else:
        has_drop = False
        
    
    epochs = input("Please state the number of epochs to train for!\n Answer: ")
    if epochs not in range(100):
        epochs = 10
        
    batch_size = input("Please state the batch size to train with!\n Answer: ")
    if batch_size not in range(500):
        batch_size = 32
    
    
    do_plot_accuracy = True
    do_plot_model = True
    
    main(number_of_categories, opt, act, has_drop, epochs, batch_size, do_plot_accuracy, do_plot_model)
    
    
    