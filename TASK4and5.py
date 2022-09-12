#!/usr/bin/env python
# coding: utf-8

# TASK 4 Multilayer Perception for Image Classification

# In[63]:


# Data Loader
import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
def gen_labels(im_name, pat1, pat2):
    if pat1 in im_name:
        label = np.array([0])
    elif pat2 in im_name:
        label = np.array([1])
    return label


# In[64]:


def get_data(data_path, data_list, img_h, img_w):
    """
    Parameters
    ----------
    train_data_path : Str
    Path to the data directory
    train_list : List
    A list containing the name of the images.
    img_h : Int
    image height to be resized to.
    img_w : Int
    image width to be resized to.
    Returns
    -------
    img_labels : Nested List
    A nested list containing the loaded images along with their
    correcponding labels.
    """
    img_labels = []
    for item in enumerate(data_list):
        img = imread(os.path.join(data_path, item[1]), as_gray=True)  # "as_grey"
    img = resize(img, (img_h, img_w), anti_aliasing=True).astype('float32')
    img_labels.append([np.array(img), gen_labels(item[1], 'Mel', 'Nev')])

    if item[0] % 100 == 0:
        print('Reading: {0}/{1} of train images'.format(item[0], len(data_list)))

    shuffle(img_labels)
    return img_labels


# In[65]:


def get_data_arrays(nested_list, img_h, img_w):
    """
     Parameters
     ----------
     nested_list : nested list
     nested list of image arrays with corresponding class labels.
     img_h : Int
     Image height.
     img_w : Int
     Image width.
     Returns
      -------
     img_arrays : Numpy array
     4D Array with the size of (n_data,img_h,img_w, 1)
     label_arrays : Numpy array
     1D array with the size (n_data).
    """
    img_arrays = np.zeros((len(nested_list), img_h, img_w), dtype=np.float32)
    label_arrays = np.zeros((len(nested_list)), dtype=np.int32)
    for ind in range(len(nested_list)):
        img_arrays[ind] = nested_list[ind][0]
        label_arrays[ind] = nested_list[ind][1]
    img_arrays = np.expand_dims(img_arrays, axis=3)
    return img_arrays, label_arrays


# In[66]:


def get_train_test_arrays(train_data_path, test_data_path, train_list,
                          test_list, img_h, img_w):
    """
    Get the directory to the train and test sets, the files names and
    the size of the image and return the image and label arrays for
    train and test sets.
    """

    train_data = get_data(train_data_path, train_list, img_h, img_w)
    test_data = get_data(test_data_path, test_list, img_h, img_w)

    train_img, train_label = get_data_arrays(train_data, img_h, img_w)
    test_img, test_label = get_data_arrays(test_data, img_h, img_w)
    del (train_data)
    del (test_data)
    return train_img, test_img, train_label, test_label


# In[67]:


img_w, img_h = 128, 128 # Setting the width and heights of the images.
data_path = '/DL_course_data/Lab1/Skin/' # Path to data root with two subdirs.
train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')
train_list = os.listdir(train_data_path)
test_list = os.listdir(test_data_path)
x_train, x_test, y_train, y_test = get_train_test_arrays(
     train_data_path, test_data_path,
     train_list, test_list, img_h, img_w)


# In[68]:


print(type(x_train), x_train.shape,x_test.shape,img_w)


# Functional API

# In[69]:


from tensorflow.keras.layers import Input, Dense, Flatten 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

def model(img_width, img_height, img_ch, base_dense):
        """
        Functional API model.
        name the last layer as "out"; e.g., out = ....
        """        
        input_size = (img_width, img_height, img_ch)
        inputs_layer = Input(shape=input_size, name='input_layer')
        inputs_layer1 = Flatten()(inputs_layer)
        # TODO
        den1 = Dense(base_dense,activation = 'relu')(inputs_layer1)
        den2 = Dense(base_dense/2,activation = 'relu')(den1)
        den3 = Dense(base_dense/2,activation = 'relu')(den2)
        out = Dense(1, activation = 'sigmoid')(den3)
        clf = Model(inputs=inputs_layer, outputs=out)
        clf.summary()
        return clf


# In[89]:


base_dense = 256#64
batchsize = 16
n_epochs = 1500  #50
LR = 0.0001  #0.1
clf =  model(img_w, img_h, 1, base_dense)

clf.compile(loss='binary_crossentropy',
              optimizer = SGD(lr = LR),
              metrics=['binary_accuracy'])


# In[90]:


print(img_w,y_train.shape,y_test.shape,x_test.shape,x_train.shape)


# In[91]:


clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data = (x_test,y_test))


# In[92]:


import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
plt.plot( np.argmin(clf_hist.history["val_loss"]),
     np.min(clf_hist.history["val_loss"]),
     marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.show()


# TASK 5 Convolutional Neural Network

# Sequential API

# In[93]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# In[94]:


def model(img_width, img_height, img_ch, base):
        """
        Functional API model.
        name the last layer as "out"; e.g., out = ....
        """        
        input_size = (img_width, img_height, img_ch)
        inputs_layer = Input(shape=input_size, name='input_layer')
        inputs_layer1 = inputs_layer
#         inputs_layer1 = Flatten()(inputs_layer)  For CNN no need to Flatten?
        conv1 = Conv2D(base, kernel_size = (3, 3), activation='relu',
                      strides=1, padding='same',
                      input_shape = (img_width, img_height, img_ch))(inputs_layer1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(base*2, kernel_size = (3, 3), activation='relu',
               strides=1, padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#         Dense layer only can handle 1D data?
        inputs_dense = Flatten()(pool2)
        hidden4 = Dense(base*2, activation='relu')(inputs_dense)
        out = Dense(1, activation='sigmoid')(hidden4)
        clf = Model(inputs=inputs_layer, outputs=out)
        clf.summary()
        return clf


# In[95]:


# def model(img_ch, img_width, img_height):
#      model = Sequential()
#      model.add(Conv2D(base, kernel_size = (3, 3), activation='relu',
#                       strides=1, padding='same',
#                       input_shape = (img_width, img_height, img_ch)))
#      model.add(MaxPooling2D(pool_size=(2, 2)))
 
#      model.add(Conv2D(base*2, kernel_size = (3, 3), activation='relu',
#                strides=1, padding='same'))
#      model.add(MaxPooling2D(pool_size=(2, 2)))
     
#      model.add(Flatten())
#      model.add(Dense(base*2, activation='relu'))
#      model.add(Dense(1, activation='sigmoid'))
#      model.summary()
#      return model


# In[102]:


firstlayer = 32
batchsize = 8
n_epochs = 200   # 20
learningrate = 0.0001   #0.00001

clf =  model(img_w, img_h, 1,firstlayer)

clf.compile(loss='binary_crossentropy',
              optimizer = SGD(lr = learningrate),
              metrics=['binary_accuracy'])


# In[103]:


clf_hist2 = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data = (x_test,y_test))


# In[104]:


import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
plt.plot( np.argmin(clf_hist.history["val_loss"]),
     np.min(clf_hist.history["val_loss"]),
     marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.show()


# In[ ]:





# In[ ]:




