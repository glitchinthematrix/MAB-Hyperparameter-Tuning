import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
import fcn 

#Preprocessing and Bifurcating  in Train and Test
"""
This block preprocesses images before they can be trained upon.
It does the followng:
- Scales the images betweeen 0 and 1
- Concatenates the HOG features to the flattened images
- Brings down the mean of the image to zero
- Dimensionality reduction using PCA. 3396 features to 1000 features
"""

data_raw = np.load('./dataset_2.npy')/255 #scaing images
labels = np.load('./dataset_2_label.npy')
hog_features = np.load('./dataset_2_hog.npy')[:,1:]
data= data_raw[1:,:]
labels = labels[:,1:]
y_train = labels[:,:9000]
y_test = labels[:,9000:]

X_train = data[:9000,:]
#X_trainm = X_train - np.mean(X_train, axis = 0) 
X_train = np.concatenate((X_train,hog_features[:,:9000].T),1) # Adding the HOG features in addition to the flattned attributes
X_trainm = X_train - np.mean(X_train, axis = 0) #Bringing mean down to 0
cov = np.dot(X_trainm.T, X_trainm) / X_trainm.shape[0] #Calculating covariance
U,S,V = np.linalg.svd(cov) #Calculating eigen vectors
X_train_uc = np.dot(X_trainm, U[:800].T) #Projecting to 1000 eigen vectors 

X_test = data[9000:,:]
#X_testm = X_test -  np.mean(X_train, axis = 0) 
X_test = np.concatenate((X_test,hog_features[:,9000:].T),1)
X_testm = X_test -  np.mean(X_train, axis = 0) 
X_test_uc = np.dot(X_testm, U[:800].T) 

#9000 train images and 1000 images in the test set
def run_config(config):
    layers = [800,config[0],config[1],2]
    discriminator = fcn.Model(layers,load=False,learning_rate=config[2]) #Initialize architecture
    discriminator.LoadData(X_train_uc,y_train.T,X_test_uc,y_test.T) #Load test and train data
    epoch_vec,train_vec,test_vec = discriminator.train(epoch=10)
    return test_vec[-1]/100

    
