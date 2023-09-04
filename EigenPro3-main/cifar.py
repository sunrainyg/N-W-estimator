# Copyright Yulu Gan 2023.

import numpy as np
import math
import torch
import torchvision
from six.moves import cPickle as pickle
import torchvision.transforms as transforms
import os
import pdb
import platform

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_data(data_loader):
    X, y = [], []
    for idx, batch in enumerate(data_loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)

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

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
    
def load():
    # Load the raw CIFAR-10 data
    transform               = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    transform_ma            = transforms.Compose(
                                # [transforms.RandomHorizontalFlip(),       # 50%的概率进行水平翻转
                                # transforms.RandomVerticalFlip(),
                                # transforms.RandomCrop(24),
                                # transforms.Resize(32),
                                # transforms.RandomRotation(50),
                                [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    trainset                = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
    trainset4ma             = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform_ma)
    testset                 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                            download=True, transform=transform)
    
    trainloader             = torch.utils.data.DataLoader(trainset, batch_size=50000,
                                                            shuffle=False, num_workers=2)
    trainloader4ma          = torch.utils.data.DataLoader(trainset4ma, batch_size=10000,
                                                            shuffle=False, num_workers=2)
    testloader              = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                            shuffle=False, num_workers=2)
    

    x_train, y_train        = get_data(trainloader)
    x_train_ma, y_train_ma  = get_data(trainloader4ma)
    x_test, y_test          = get_data(testloader)
    
    x_train                 = x_train.reshape(x_train.shape[0],-1)
    x_train_ma              = x_train_ma.reshape(x_train_ma.shape[0],-1)
    x_test                  = x_test.reshape(x_test.shape[0],-1)
    
    y_train                 = to_categorical(y_train)
    y_test                  = to_categorical(y_test)
    y_train_ma              = to_categorical(y_train_ma)

    # x_train, y_train        = torch.cat((x_train, x_train_ma[0:20000]), dim=0), np.concatenate((y_train, y_train_ma[0:20000]), axis=0)
    
    x_train, y_train, x_test, y_test = x_train.numpy(), y_train, x_test.numpy(), y_test
    return (x_train, y_train), (x_test, y_test), (x_train_ma, y_train_ma)


if __name__ == "__main__":

    cifar10_dir = '/Users/yulu/N-W-estimator/dataset/cifar10/cifar-10-batches-py'
    # x_train, y_train, x_val, y_val, x_test, y_test = load(cifar10_dir)
    (x_train, y_train), (x_test, y_test) = load(cifar10_dir)