# -*- coding: utf-8 -*-
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""  Tom Humbert                                                 """
"""  CS Project - Image Classification                           """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import pickle, numpy
import tensorflow as tf



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_full_ds(folder): 
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    for file in os.listdir(folder):
        if "data" in file:
            datadict = unpickle(folder+"//"+file)
            if len(train_images) < 1:
                train_images = numpy.asarray(datadict[b'data'])
                train_labels = datadict[b'labels']
                
            else:
                train_images = numpy.append(train_images, datadict[b'data'], axis=0)
                train_labels += datadict[b'labels']
            
        elif "test" in file:
            datadict = unpickle(folder+"//"+file)
            test_images = datadict[b'data']
            test_labels = datadict[b'labels']

    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    return train_images, train_labels, test_images, test_labels

    
def get_categories(data, labels, cat_list = 0):
    """
    This function takes as parameter a list of category names and returns a 
    dictionary containing only data from these categories
    """
    categories = {"airplane":0, "automobile":1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    
    if cat_list == 0 or len(cat_list)<1:
        return f"The categories to choose from are {categories.keys()}"
    
    #filter out n first categories
    else:
        cats = []
        for cat in cat_list:
            cats.append(categories[cat])
        
        newlabels = []
        newdata = []
        for i in range(len(labels)):
            if labels[i] in cats:
                newlabels.append(labels[i])
                newdata.append(data[i])

        newdata, newlabels = tf.stack(newdata), tf.stack(newlabels)
    """
    # The code below works, but is incredibly slow. So I am using Ciarans method (above).
    else:
        cats = []
        new_data = []
        new_lbls = []
        for cat in cat_list:
            cats.append(categories[cat])
    
        for i in range(len(labels)):
            if labels[i] in cats:
                new_lbls.append(labels[i])
                
                if len(new_data)<1:
                    new_data = numpy.asarray([data[i]])
                
                else:
                    new_data = numpy.append(new_data, [data[i]], axis=0)

        return new_lbls, new_data
    """ 
    return newlabels, newdata
            
    
    
def stats(imgs, lbls):
    import pandas as pd
  
    df = pd.DataFrame(lbls)
    return df[0].value_counts()


def check_for_duplicates_and_remove(train_images, train_labels, test_images, test_labels):

    for i in range(len(train_labels)-1, 0, -1):
        for j in range(len(test_labels)):
            if bool(tf.reduce_all(tf.equal(train_imgs[i], test_imgs[j]))):
                print(i,  j)
            
    return train_images, train_labels
    
def count_occurences_of_categories(lbls):
    occ_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    for occ in lbls:
        occ_dict[occ] += 1
        
    return occ_dict
    
if __name__ == "__main__":
    # Retrieving full dataset
    train_images, train_labels, test_images, test_labels = get_full_ds("..\\cifar-10-python\\cifar-10-batches-py")
    
    # Testing the selection process
    train_lbls, train_imgs = get_categories(train_images, train_labels, cat_list = ['airplane', 'dog', 'bird', 'cat', 'ship'])
    test_lbls, test_imgs = get_categories(test_images, test_labels, cat_list = ['airplane', 'dog', 'bird', 'cat', 'ship'])
    print(len(train_lbls))
    print(train_imgs.shape)
    
    # Check if the testing and training set are mutually exclusive, else remove duplicates
    train_imgs, train_lbls = check_for_duplicates_and_remove(train_imgs, train_lbls, test_imgs, test_lbls)
    
    # I apparantly created two functions to do the same thing so lets check if the count of occurences for both is same.
    count_train = stats(train_imgs, train_lbls)
    count_test = stats(test_imgs, test_lbls)
    
    count2_train = count_occurences_of_categories(train_lbls)
    
    print(count_train)
    print(count2_train)
    
    
