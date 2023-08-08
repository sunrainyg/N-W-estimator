# Copyright Yulu Gan 2023.

import numpy as np
import torchvision.transforms as transforms
import torchvision, torch
import pdb

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

def unit_range_normalize(samples):
    '''
    FUNCTION: Do normalization
    Input: samples
    Output: normalized samples
    '''
    min_vals = np.min(samples, axis=0)
    max_vals = np.max(samples, axis=0)
    diff = max_vals - min_vals
    diff[diff <= 0.0] = np.maximum(1.0, min_vals[diff <= 0.0])
    normalized = (samples - min_vals) / diff
    return normalized

def load():
    '''
    Output: x_train: (60000, 28*28*1)
            y_train: (60000, 10)
            x_test: (10000, 28*28*1)
            y_test: (10000, 10)
    Output format: <class 'numpy.ndarray'>
    '''
    
    # input image dimensions
    n_class = 10
    img_rows, img_cols = 28, 28
    # the data, shuffled and split between train and test sets
    
    # (x_train, y_train), (x_test, y_test) = load_data()
    
    path = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/data/mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train'] # y_train:(60000,)
    x_test, y_test = f['x_test'], f['y_test'] # y_test:(60000,)
    f.close()
    
    x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    x_train = unit_range_normalize(x_train) # (60000, 784)
    x_test = unit_range_normalize(x_test)
    y_train = to_categorical(y_train, n_class) # label -> one hot
    y_test = to_categorical(y_test, n_class)
    print("Load MNIST dataset.")
    
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    return (x_train, y_train), (x_test, y_test)


def load_10classes():
    
    n_class = 10
    
    transform       = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))])
    
    transform_ma    = transforms.Compose(
                        # [transforms.RandomHorizontalFlip(),       # 50%的概率进行水平翻转
                        # transforms.RandomVerticalFlip(),
                        # transforms.RandomRotation(50),
                        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.1),
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))])
    
    trainset                  = torchvision.datasets.MNIST(root='./data', train=True,
                                                            download=True, transform=transform)
    testset                   = torchvision.datasets.MNIST(root='./data', train=False,
                                                            download=True, transform=transform)
    trainset4ma               = torchvision.datasets.MNIST(root='./data', train=True,
                                                            download=True, transform=transform_ma)
    trainloader               = torch.utils.data.DataLoader(trainset, batch_size=60000,
                                                            shuffle=False, num_workers=2)
    testloader                = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                            shuffle=False, num_workers=2)
    trainloader4ma            = torch.utils.data.DataLoader(trainset4ma, batch_size=60000,
                                                            shuffle=False, num_workers=2)
    
    for step, (train_batch_x, train_batch_y) in enumerate(trainloader):
        if step == 1:
            break
        x_train     = train_batch_x.view(60000, -1)
        y_train_ori = train_batch_y
        y_train     = to_categorical(y_train_ori, n_class) # label -> one hot
        
    for step, (train_batch_x4ma, train_batch_y4ma) in enumerate(trainloader4ma):
        if step == 1:
            break
        x_train4ma = train_batch_x4ma.view(60000, -1)
        ### combine data
        x_train4ma = torch.cat((x_train, x_train4ma), dim=0)
        
        y_train4ma = train_batch_y4ma
        y_train4ma = torch.cat((y_train_ori, y_train4ma), dim=0)
        y_train4ma = to_categorical(y_train4ma, n_class) # label -> one hot
        
    for step, (test_batch_x, test_batch_y) in enumerate(testloader):
        x_test = test_batch_x.view(10000, -1)
        y_test = test_batch_y
        y_test = to_categorical(y_test, n_class) # label -> one hot
    
    
    return (x_train, y_train), (x_test, y_test), trainset