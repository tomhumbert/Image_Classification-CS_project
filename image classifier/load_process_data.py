# -*- coding: utf-8 -*-
import sys, os
import pickle, numpy



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

    
def get_categories(data, labels, cat_list = 0):
    """
    This function takes as parameter a list of category names and returns a 
    dictionary containing only data from these categories
    """
    categories = {"airplane":0, "automobile":1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    
    if cat_list == 0 or len(cat_list)<1:
        return f"The categories to choose from are {categories.keys()}"
    
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
            
    
    
def stats(imgs, lbls):
    import pandas as pd
  
    df = pd.DataFrame(lbls)
    print("These are the counts of the labels in the given set")
    print(df[0].value_counts())
    
    
    #df.plot()
    #plt.show()
    
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
    
    
if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = get_full_ds("..\\cifar-10-python\\cifar-10-batches-py")
    stats(train_images, train_labels)
    stats(test_images, test_labels)
    sel_lbl, sel_img = get_categories(train_images, train_labels, ['airplane', 'cat'])
    stats(sel_img, sel_lbl)
