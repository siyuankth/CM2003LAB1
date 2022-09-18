#!/usr/bin/env python
# coding: utf-8

# # Loading data 

# In[33]:


# Data Loader
import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize


# In[34]:


def gen_labels(im_name, pat1, pat2):
    '''
    Parameters
    ----------
    im_name : Str
        The image file name.
    pat1 : Str
        A string pattern in the filename for 1st class, e.g "Mel"
    pat2 : Str
        A string pattern in the filename 2nd class, e.g, "Nev" 
    Returns
    -------
    Label : Numpy array        
        Class label of the filename name based on its pattern.
    '''
    if pat1 in im_name:
        label = np.array([0])
    elif pat2 in im_name:
        label = np.array([1])
    return label


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
        img = imread(os.path.join(data_path, item[1]), as_gray = True) # "as_grey"
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        img_labels.append([np.array(img), gen_labels(item[1], 'Mel', 'Nev')]) 
        
        if item[0] % 100 == 0:
             print('Reading: {0}/{1}  of train images'.format(item[0], len(data_list)))
             
    shuffle(img_labels)
    return img_labels


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
    img_arrays = np.zeros((len(nested_list), img_h, img_w), dtype = np.float32)
    label_arrays = np.zeros((len(nested_list)), dtype = np.int32)
    for ind in range(len(nested_list)):
        img_arrays[ind] = nested_list[ind][0]
        label_arrays[ind] = nested_list[ind][1]
    img_arrays = np.expand_dims(img_arrays, axis =3)
    return img_arrays, label_arrays


def get_train_test_arrays(train_data_path, test_data_path, train_list,
                          test_list, img_h, img_w):
    """
    Get the directory to the train and test sets, the files names and
    the size of the image and return the image and label arrays for
    train and test sets.
    """
    
    train_data = get_data(train_data_path, train_list, img_h, img_w)
    test_data = get_data(test_data_path, test_list, img_h, img_w)
    
    train_img, train_label =  get_data_arrays(train_data, img_h, img_w)
    test_img, test_label = get_data_arrays(test_data, img_h, img_w)
    del(train_data)
    del(test_data)      
    return train_img, test_img, train_label, test_label


# In[46]:


img_w, img_h = 128, 128      # Setting the width and heights of the images.
data_path = '/DL_course_data/Lab1/Skin/'     # Path to data root with two subdirs.
train_data_path = os.path.join(data_path, 'train')   
test_data_path = os.path.join(data_path, 'test')
train_list = os.listdir(train_data_path)
test_list = os.listdir(test_data_path)
x_train, x_test, y_train, y_test = get_train_test_arrays(
        train_data_path, test_data_path,
        train_list, test_list, img_h, img_w)


# # Task 6A

# In[36]:


# Logic operator with Tensorflow Keras and importation of matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


# In[37]:


def model(img_ch, img_width, img_height, n_base):
    model = Sequential()
    
    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model


# In[6]:


img_w, img_h = 128, 128
base = 32
n_epochs = 50
batchsize = 8
LR = 0.0001

clf = model(1, img_w, img_h, base)


# In[7]:


clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])


# In[8]:


clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))


# In[12]:


import matplotlib.pyplot as plt
#Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('6A/loss_32base.png', dpi = 200)

#Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
ymax = np.max(clf_hist.history["val_binary_accuracy"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('6A/acc_32base.png', dpi = 200)


# ### _Comments_
# The model fit quite well the training model, because at the last iteration, the loss value is really close to 0. But the validation dataset is not well fitted and it is worse through the iterations, this shows a clear overfitting.

# # Task 6B

# In[13]:


def model_dropout(img_ch, img_width, img_height, n_base):
    model = Sequential()
    
    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0,4))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model


# In[14]:


# Without dropout

img_w, img_h = 128, 128
n_epochs = 50
batchsize = 8
LR = 0.0001

for base in [8,16]:
    #Building the models
    clf = model(1, img_w, img_h, base)
    clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
    clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
    
    #Plotting curves
    #Loss
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(clf_hist.history["loss"], label="loss")
    plt.plot(clf_hist.history["val_loss"], label="val_loss")
    xmin = np.argmin(clf_hist.history["val_loss"])
    ymin = np.min(clf_hist.history["val_loss"])
    plt.plot( xmin, ymin, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
                 horizontalalignment = "center", verticalalignment = "top", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('6B/loss_'+str(base)+'.png', dpi = 200)

    #Accuracy
    plt.figure(figsize=(4, 4))
    plt.title("Accuracy")
    plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
    plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
    xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
    ymax = np.max(clf_hist.history["val_binary_accuracy"])
    plt.plot( xmax, ymax, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
                 horizontalalignment = "center", verticalalignment = "bottom", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('6B/acc_'+str(base)+'.png', dpi = 200)


# In[15]:


# With dropout

img_w, img_h = 128, 128
batchsize = 8
LR = 0.0001
base = 8

for n_epochs in [50, 150]:
    #Building the models
    clf = model_dropout(1, img_w, img_h, base)
    clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
    clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
    
    #Plotting curves
    #Loss
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(clf_hist.history["loss"], label="loss")
    plt.plot(clf_hist.history["val_loss"], label="val_loss")
    xmin = np.argmin(clf_hist.history["val_loss"])
    ymin = np.min(clf_hist.history["val_loss"])
    plt.plot( xmin, ymin, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
                 horizontalalignment = "center", verticalalignment = "top", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('6B/loss_'+str(n_epochs)+'_with_dropout.png', dpi = 200)

    #Accuracy
    plt.figure(figsize=(4, 4))
    plt.title("Accuracy")
    plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
    plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
    xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
    ymax = np.max(clf_hist.history["val_binary_accuracy"])
    plt.plot( xmax, ymax, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
                 horizontalalignment = "center", verticalalignment = "bottom", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('6B/acc_'+str(n_epochs)+'_with_dropout.png', dpi = 200)
    


# ### _Comments_
# Reducing the n_base reduces the overfitting but does not cancel it. It is normal that it reduces it because there are less neuron and it will fit less the noise of the training dataset. 
# As expected the dropout layer reduces also the overfitting, however it doesn't cancel it either. This reduce is expected because the drop out layer randomly delete some neurons in the dense layer, making the learning power of the network weaker and therefore it cannot fit too well the training dataset.

# # Tak 6C

# In[16]:


img_w, img_h = 128, 128
base = 8
batchsize = 8
LR = 1e-5
for n_epochs in [150, 350]:
    #Building the models
    clf = model(1, img_w, img_h, base)
    clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
    clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
    
    #Plotting curves
    #Loss
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(clf_hist.history["loss"], label="loss")
    plt.plot(clf_hist.history["val_loss"], label="val_loss")
    xmin = np.argmin(clf_hist.history["val_loss"])
    ymin = np.min(clf_hist.history["val_loss"])
    plt.plot( xmin, ymin, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
                 horizontalalignment = "center", verticalalignment = "top", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('6C/loss_'+str(n_epochs)+'.png', dpi = 200)

    #Accuracy
    plt.figure(figsize=(4, 4))
    plt.title("Accuracy")
    plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
    plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
    xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
    ymax = np.max(clf_hist.history["val_binary_accuracy"])
    plt.plot( xmax, ymax, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
                 horizontalalignment = "center", verticalalignment = "bottom", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('6C/acc_'+str(n_epochs)+'.png', dpi = 200)


# ### _Comments_
# blabla

# # Task 6D

# In[17]:


img_w, img_h = 128, 128
base = 8
n_epochs = 150
LR = 1e-5
for batchsize in [2,4,8]:
    #Building the model
    clf = model(1, img_w, img_h, base)
    clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
    clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
    
    #Loss
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(clf_hist.history["loss"], label="loss")
    plt.plot(clf_hist.history["val_loss"], label="val_loss")
    xmin = np.argmin(clf_hist.history["val_loss"])
    ymin = np.min(clf_hist.history["val_loss"])
    plt.plot( xmin, ymin, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
                 horizontalalignment = "center", verticalalignment = "top", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('6D/loss_batchsize_'+str(batchsize)+'.png', dpi = 200)
    
    #Accuracy
    plt.figure(figsize=(4, 4))
    plt.title("Accuracy")
    plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
    plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
    xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
    ymax = np.max(clf_hist.history["val_binary_accuracy"])
    plt.plot( xmax, ymax, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
                 horizontalalignment = "center", verticalalignment = "bottom", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('6D/acc_batchsize_'+str(batchsize)+'.png', dpi = 200)


# # Task 6E

# #### _Finding the porper learning rate_

# In[24]:


img_w, img_h = 128, 128
base = 32
batchsize = 8
n_epochs = 150
for LR in [1e-3, 1e-4, 1e-5]:
    #Building the model
    clf = model(1, img_w, img_h, base)
    clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
    clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
    
    #Loss
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(clf_hist.history["loss"], label="loss")
    plt.plot(clf_hist.history["val_loss"], label="val_loss")
    xmin = np.argmin(clf_hist.history["val_loss"])
    ymin = np.min(clf_hist.history["val_loss"])
    plt.plot( xmin, ymin, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
                 horizontalalignment = "center", verticalalignment = "top", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('6E/loss_lr_'+str(LR)+'.png', dpi = 200)
    
    #Accuracy
    plt.figure(figsize=(4, 4))
    plt.title("Accuracy")
    plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
    plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
    xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
    ymax = np.max(clf_hist.history["val_binary_accuracy"])
    plt.plot( xmax, ymax, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
                 horizontalalignment = "center", verticalalignment = "bottom", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('6E/acc_lr_'+str(LR)+'.png', dpi = 200)


# In[32]:


img_w, img_h = 128, 128
base = 8
batchsize = 4
n_epochs = 100
LR = 1e-5
#Building the model
clf = model(1, img_w, img_h, base)
clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))  

#Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('6E/loss_bestmodel.png', dpi = 200)
 
#Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
ymax = np.max(clf_hist.history["val_binary_accuracy"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('6E/acc_bestmodel.png', dpi = 200)


# In[52]:


img_w, img_h = 128, 128
base = 8
batchsize = 4
n_epochs = 100
LR = 1e-5
#Building the model
clf = model(1, img_w, img_h, base)
clf.compile(loss='binary_crossentropy', optimizer = SGD(lr = LR), metrics=['binary_accuracy'])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))  

#Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('6E/loss_bestmodel_SGD.png', dpi = 200)
 
#Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
ymax = np.max(clf_hist.history["val_binary_accuracy"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('6E/acc_bestmodel_SGD.png', dpi = 200)


# In[39]:


img_w, img_h = 128, 128
base = 8
batchsize = 4
n_epochs = 100
LR = 1e-5
#Building the model
clf = model(1, img_w, img_h, base)
clf.compile(loss='binary_crossentropy', optimizer = RMSprop(lr = LR), metrics=['binary_accuracy'])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))  

#Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('6E/loss_bestmodel_rmsprop.png', dpi = 200)
 
#Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
ymax = np.max(clf_hist.history["val_binary_accuracy"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('6E/acc_bestmodel_rmsprop.png', dpi = 200)


# # Task 6F 

# In[40]:


img_w, img_h = 128, 128
base = 8
batchsize = 4
n_epochs = 100
LR = 1e-5

#Prepare the y dataset
y_test[y_test == 0] = -1
y_train[y_train == 0] = -1

#Building the model
clf = model(1, img_w, img_h, base)
clf.compile(loss='hinge', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))  

#Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('6F/loss_hinge.png', dpi = 200)
 
#Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
ymax = np.max(clf_hist.history["val_binary_accuracy"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('6F/acc_hinge.png', dpi = 200)


# # Task 7A

# In[42]:


# Logic operator with Tensorflow Keras and importation of matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


# In[43]:


def model_vgg16(img_ch, img_width, img_height, n_base):
    model = Sequential()
    
    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model


# # Task 7B

# In[47]:


img_w, img_h = 128, 128
base = 8
batchsize = 8
n_epochs = 100
for LR in [1e-3, 1e-4, 1e-5]:
    #Building the model
    clf = model_vgg16(1, img_w, img_h, base)
    clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
    clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
    
    #Loss
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(clf_hist.history["loss"], label="loss")
    plt.plot(clf_hist.history["val_loss"], label="val_loss")
    xmin = np.argmin(clf_hist.history["val_loss"])
    ymin = np.min(clf_hist.history["val_loss"])
    plt.plot( xmin, ymin, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
                 horizontalalignment = "center", verticalalignment = "top", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('7B/loss_lr_'+str(LR)+'.png', dpi = 200)
    
    #Accuracy
    plt.figure(figsize=(4, 4))
    plt.title("Accuracy")
    plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
    plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
    xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
    ymax = np.max(clf_hist.history["val_binary_accuracy"])
    plt.plot( xmax, ymax, marker="x", color="r", label="best model")
    plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
                 horizontalalignment = "center", verticalalignment = "bottom", color = "red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig('7B/acc_lr_'+str(LR)+'.png', dpi = 200)


# # Task 7C

# In[48]:


img_w, img_h = 128, 128
base = 8
batchsize = 8
LR = 1e-5
n_epochs = 150

#Building the models
clf = model_vgg16(1, img_w, img_h, base)
clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
    
#Plotting curves
#Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('7C/loss_'+str(n_epochs)+'.png', dpi = 200)

#Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
ymax = np.max(clf_hist.history["val_binary_accuracy"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('7C/acc_'+str(n_epochs)+'.png', dpi = 200)


# # Task 7D

# In[49]:


img_w, img_h = 128, 128
base = 16
batchsize = 8
LR = 1e-5
n_epochs = 100

#Building the models
clf = model_vgg16(1, img_w, img_h, base)
clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
    
#Plotting curves
#Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('7D/loss.png', dpi = 200)

#Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
ymax = np.max(clf_hist.history["val_binary_accuracy"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('7D/acc.png', dpi = 200)


# In[50]:


def model_vgg16_dropout(img_ch, img_width, img_height, n_base):
    model = Sequential()
    
    model.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch),kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model


# In[51]:


img_w, img_h = 128, 128
base = 16
batchsize = 8
LR = 1e-5
n_epochs = 100

#Building the models
clf = model_vgg16_dropout(1, img_w, img_h, base)
clf.compile(loss='binary_crossentropy', optimizer = Adam(lr = LR), metrics=['binary_accuracy'])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
    
#Plotting curves
#Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('7D/loss_dropout.png', dpi = 200)

#Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["binary_accuracy"], label="accuracy")
plt.plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
xmax = np.argmax(clf_hist.history["val_binary_accuracy"])
ymax = np.max(clf_hist.history["val_binary_accuracy"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig('7D/acc_dropout.png', dpi = 200)


# In[56]:


import os
data_path = '/DL_course_data/Lab1/X_ray'

test_idx = 20

train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')

test_list = os.listdir(test_data_path)
filename = os.path.join(test_data_path, test_list[test_idx])


# In[59]:



img_ch = 1
img = imread(filename, as_gray = True)
img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
pred_class = clf.predict(np.reshape(img, (1,img_w, img_h, img_ch)))
plt.figure(figsize = (6,6))
plt.imshow(img, cmap = 'gray')
plt.show()
print('Test file : ', filename)
print('True Class : ', class_list[gen_labels(filename, class_list)])
print('Predicted Class : ', class_list[np.argmax(pred_class)])

