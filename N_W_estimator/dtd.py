# Copyright Yulu Gan 2023.

import numpy as np
import math
from six.moves import cPickle as pickle
from robustbench.data import load_cifar10c
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
import torch
import torchvision
import os
import cv2
import pdb
import platform

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch

def to_categorical_tensor(y, num_classes=None, dtype=torch.float32):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a PyTorch dtype
            (e.g., torch.float32, torch.float64, torch.int32, ...).

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = torch.tensor(y, dtype=torch.int32)
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.view(-1)
    if not num_classes:
        num_classes = torch.max(y) + 1
    n = y.shape[0]
    categorical = torch.zeros((n, num_classes), dtype=dtype)

    categorical[torch.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = categorical.view(output_shape)
    return categorical



def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)

    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def get_data(data_loader):
    X, y = [], []
    for idx, batch in enumerate(data_loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)

def pre_process(torchset,n_samples,num_classes=10):
    indices = list(np.random.choice(len(torchset),n_samples))

    trainset = []
    for ix in indices:
        x,y = torchset[ix]
        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        trainset.append(((x/np.linalg.norm(x)).reshape(-1),ohe_y))
    return trainset
    
def load_10classes():
    
    n_class = 10
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    trainset0 = torchvision.datasets.DTD(root='./data',
                                        split = "train",
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.DTD(root='./data',
                                        split = "test",
                                        transform=transform,
                                        download=True)

    trainset = pre_process(trainset0,n_samples=20000, num_classes=10)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)


    testset = pre_process(testset0,n_samples=5000, num_classes=10)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=2)

    x_train, y_train = get_data(train_loader) #y_train: torch.Size([20000, 10])
    x_test, y_test = get_data(test_loader)
    return (x_train, y_train), (x_test, y_test)