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
    
def load_10classes(cifar10_dir):
    
    n_class = 10
    
    transform       = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    transform_vis   = transforms.Compose(
                        [transforms.ToTensor()])
    
    transform_ma    = transforms.Compose(
                        # [transforms.RandomHorizontalFlip(),       # 50%的概率进行水平翻转
                        # transforms.RandomVerticalFlip(),
                        # transforms.RandomRotation(50),
                        [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    trainset                  = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                            download=True, transform=transform)
    
    trainset_vis              = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                            download=True, transform=transform_vis)
    
    testset                   = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                            download=True, transform=transform)
    trainset4ma               = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                            download=True, transform=transform_ma)
    
    trainset                  = pre_process(trainset, n_samples=50000, num_classes=10)
    trainloader               = torch.utils.data.DataLoader(trainset, batch_size=30000,
                                                            shuffle=False, num_workers=2)
    testset                   = pre_process(testset, n_samples=50000, num_classes=10)
    testloader                = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                            shuffle=False, num_workers=2)
    trainset4ma               = pre_process(trainset4ma, n_samples=50000, num_classes=10)
    trainloader4ma            = torch.utils.data.DataLoader(trainset4ma, batch_size=30000,
                                                            shuffle=False, num_workers=2)
    
    x_train, y_train          = get_data(trainloader) #y_train: torch.Size([20000, 10])
    x_test, y_test            = get_data(testloader)
    x_train4ma, y_train4ma    = get_data(trainloader4ma)
    x_train4ma, y_train4ma    = torch.cat((x_train, x_train4ma), dim=0), torch.cat((y_train, y_train4ma), dim=0)
    
    # for step, (train_batch_x, train_batch_y) in enumerate(trainloader):
    #     if step == 1:
    #         break
    #     x_train     = train_batch_x.view(30000, -1)
    #     y_train_ori = train_batch_y
    #     y_train     = to_categorical(y_train_ori, n_class) # label -> one hot
        
    # for step, (train_batch_x4ma, train_batch_y4ma) in enumerate(trainloader4ma):
    #     if step == 1:
    #         break
    #     x_train4ma = train_batch_x4ma.view(30000, -1)
    #     ### combine data
    #     x_train4ma = torch.cat((x_train, x_train4ma), dim=0)
        
    #     y_train4ma = train_batch_y4ma
    #     y_train4ma = torch.cat((y_train_ori, y_train4ma), dim=0)
    #     y_train4ma = to_categorical(y_train4ma, n_class) # label -> one hot
        
    # for step, (test_batch_x, test_batch_y) in enumerate(testloader):
    #     x_test = test_batch_x.view(10000, -1)
    #     y_test = test_batch_y
    #     y_test = to_categorical(y_test, n_class) # label -> one hot
    
    ### !!for corruptions!! if not use, please remove it
    # x_test_brightness, y_test_brightness    = load_cifar10c(10000, 1, './data/', False,
    #                                        ['brightness'])
    # x_test_contrast, y_test_contrast        = load_cifar10c(10000, 1, './data/', False,
    #                                        ['contrast'])
    # x_test_motion_blur, y_test_motion_blur  = load_cifar10c(10000, 1, './data/', False,
    #                                        ['motion_blur'])
    # x_test_brightness                       = x_test_brightness.reshape(10000, -1)
    # x_test_contrast                         = x_test_contrast.reshape(10000, -1)
    # x_test_motion_blur                      = x_test_motion_blur.reshape(10000, -1)
    
    # x_train4ma                              = torch.cat((x_train, x_test_brightness, x_test_contrast, x_test_motion_blur), dim=0)
    # y_train4ma                              = torch.cat((y_train_ori, y_test_brightness, y_test_contrast, y_test_motion_blur), dim=0)
    # y_train4ma                              = to_categorical(y_train4ma, n_class) # label -> one hot
    
    return (x_train, y_train), (x_test, y_test), (x_train4ma, y_train4ma), trainset_vis
    ############

def load_2classes_ori(cifar10_dir, num_training=49000, num_validation=1000, num_test=10000):
    n_class = 2
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # 制作mask, 类别为1和类别为2的，mask里会置为1
    mask_class_train_1      = y_train == 0 # 指定类别
    mask_class_train_2      = y_train == 1 # 指定类别
    mask_class_test_1       = y_test  == 0
    mask_class_test_2       = y_test  == 1
    
    mask_class_train_1      = mask_class_train_1.astype(int)
    mask_class_train_2      = mask_class_train_2.astype(int)
    mask_class_test_1       = mask_class_test_1.astype(int)
    mask_class_test_2       = mask_class_test_2.astype(int)
    
    mask_class_train        = np.bitwise_or(mask_class_train_1, mask_class_train_2)
    mask_class_test         = np.bitwise_or(mask_class_test_1, mask_class_test_2)
    
    # 制作数据，二分类，类别1和类别2，训练集1w张，测试集2000张
    X_train_2clses          = X_train[mask_class_train==1]
    Y_train_2clses          = y_train[mask_class_train==1]
    X_test_2clses           = X_test[mask_class_test==1]
    Y_test_2clses           = y_test[mask_class_test==1]

    x_train = X_train_2clses.astype('float32')
    x_test = X_test_2clses.astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    y_train = to_categorical(Y_train_2clses, n_class) # label -> one hot
    y_test = to_categorical(Y_test_2clses, n_class)
    
    
    # pdb.set_trace()
    return (x_train, y_train), (x_test, y_test)

def load_2classes(cifar10_dir):
    n_class = 2
    
    transform       = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    transform_ma    = transforms.Compose(
                        # [transforms.RandomHorizontalFlip(),       # 50%的概率进行水平翻转
                        # transforms.RandomVerticalFlip(),
                        # transforms.RandomRotation(50),
                        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.1),
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    trainset                  = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                            download=True, transform=transform)
    testset                   = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                            download=True, transform=transform)
    trainset4ma               = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                            download=True, transform=transform_ma)
    trainloader               = torch.utils.data.DataLoader(trainset, batch_size=50000,
                                                            shuffle=False, num_workers=2)
    testloader                = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                            shuffle=False, num_workers=2)
    trainloader4ma            = torch.utils.data.DataLoader(trainset4ma, batch_size=50000,
                                                            shuffle=False, num_workers=2)
    
    
    for step, (train_batch_x, train_batch_y) in enumerate(trainloader):
        if step == 1:
            break
        x_train                 = train_batch_x.view(50000, -1)
        y_train_ori             = train_batch_y
        
        mask_class_train_1      = y_train_ori == 0 # 指定类别
        mask_class_train_2      = y_train_ori == 1 # 指定类别
        mask_class_train_1      = mask_class_train_1.int()
        mask_class_train_2      = mask_class_train_2.int()
        
        mask_class_train        = np.bitwise_or(mask_class_train_1, mask_class_train_2)
        x_train                 = x_train[mask_class_train==1]
        Y_train_2clses_         = y_train_ori[mask_class_train==1]

        y_train     = to_categorical(Y_train_2clses_, n_class) # label -> one hot

    for step, (train_batch_x4ma, train_batch_y4ma) in enumerate(trainloader4ma):
        if step == 1:
            break
        x_train4ma = train_batch_x4ma.view(50000, -1)
        
        mask_class_train_1      = train_batch_y4ma == 0 # 指定类别
        mask_class_train_2      = train_batch_y4ma == 1 # 指定类别
        mask_class_train_1      = mask_class_train_1.int()
        mask_class_train_2      = mask_class_train_2.int()
        
        mask_class_train        = np.bitwise_or(mask_class_train_1, mask_class_train_2)
        x_train4ma              = x_train4ma[mask_class_train==1]
        
        ### combine data
        x_train4ma              = np.concatenate((x_train, x_train4ma), axis=0)
        
        y_train4ma              = train_batch_y4ma
        Y_train_2clses          = y_train4ma[mask_class_train==1]
        y_train4ma              = np.concatenate((Y_train_2clses_, Y_train_2clses), axis=0)
        y_train4ma              = to_categorical(y_train4ma, n_class) # label -> one hot
        
    for step, (test_batch_x, test_batch_y) in enumerate(testloader):
        
        x_test                  = test_batch_x.view(10000, -1)
        
        mask_class_train_1      = test_batch_y == 0 # 指定类别
        mask_class_train_2      = test_batch_y == 1 # 指定类别
        mask_class_train_1      = mask_class_train_1.int()
        mask_class_train_2      = mask_class_train_2.int()
        
        mask_class_train        = np.bitwise_or(mask_class_train_1, mask_class_train_2)
        x_test                  = x_test[mask_class_train==1]
        y_test                  = test_batch_y[mask_class_train==1]
        y_test                  = to_categorical(y_test, n_class) # label -> one hot
    
    
    return (x_train, y_train), (x_test, y_test), (x_train4ma, y_train4ma)

if __name__ == "__main__":

    cifar10_dir = '/Users/yulu/N-W-estimator/dataset/cifar10/cifar-10-batches-py'
    # x_train, y_train, x_val, y_val, x_test, y_test = load(cifar10_dir)
    (x_train, y_train), (x_test, y_test) = load(cifar10_dir)