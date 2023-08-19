# Copyright Yulu Gan 2023.

import numpy as np
import math
from six.moves import cPickle as pickle
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
import torch
import torchvision
import os
import cv2
import pdb
import platform
import torch
import queue


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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


class udfNetWork:
    def __init__(self, batch_size, lr = 0.01):

        self.W = torch.eye(3072)

        self.t = torch.empty(batch_size, 3072)
        torch.nn.init.uniform_(self.t, a=0, b=0.05)

        #self.t = torch.randn(size=(batch_size, 3072))

        self.c = torch.empty(batch_size, 10)
        torch.nn.init.constant_(self.c, 0.1)

        self.L = 0.1

        self.learning_rate = lr

    def loss_function(self, y_pred, y_true, lambda_s = 0.0):

        mse = (torch.tensor(y_true) - y_pred).pow(2)

        l2_norm = self.W.pow(2).mean() + self.t.pow(2).mean() + self.c.pow(2).mean()

        loss = mse + lambda_s * l2_norm

        return loss

    def compute_deviation(self, y_pred, y_true):
        deviation = torch.tensor(y_true) - y_pred
        deviation = torch.sum(deviation, dim=1)
        return deviation

    def G_functio(self, x):

        xt = x - self.t
        x = xt @ self.W.T @ self.W @ xt.T
        x.clamp_(min=0)

        g_out = torch.exp(-x/self.L)
        return g_out

    def laplacian_M(self, x):
        '''
        Equation: exp(\gamma * (x-xi)M(x-xi)^T)
        '''
     
        bandwidth = self.L
        centers = self.t

        W_T = self.W.transpose(0,1)
        M = W_T @ self.W

        assert bandwidth > 0
        kernel_mat = self.euclidean_distances_M(x, centers, M, squared=False)
        kernel_mat.clamp_(min=0) # Guaranteed non-negative
        gamma = 1. / bandwidth
        kernel_mat.mul_(-gamma) # point-wise multiply
        kernel_mat.exp_() #point-wise exp
        return kernel_mat
    
    def euclidean_distances_M(self, samples, centers, M, squared=True):
        '''
        Calculate the Euclidean Distances between the samples and centers, using Ma distance
        '''
        
        if len(M.shape)==1:
            return euclidean_distances_M_diag(samples, centers, M, squared=squared)
        
        ## Obtain a vector containing the square norm value for each sample point.
        samples_norm2 = ((samples @ M) * samples).sum(-1)
    
        if samples is centers:
            centers_norm2 = samples_norm2
        else:
            centers_norm2 = ((centers @ M) * centers).sum(-1)
    
        distances = -2 * (samples @ M) @ centers.T
        distances.add_(samples_norm2.view(-1, 1))
        distances.add_(centers_norm2)
    
        if not squared:
            distances.clamp_(min=0).sqrt_()
    
        return distances

    def forward(self, x):

        g_out = self.G_functio(x)

        y = g_out @ self.c
        
        return y

    def update(self, x, y):

        y_pred = self.forward(x)
        #print('y_pred: ', y_pred.shape)
        delta = self.loss_function(y_pred, y)
        
        #print('delta: ', delta.shape)

        g_out = self.G_functio(x)
        grad_c = -2 * (g_out @ delta).sum(0)
        self.c = self.c - self.learning_rate * grad_c

        xt = x - self.t

        M = self.W.T @ self.W

        grad_t = 4 * (self.c @ delta.T @ g_out @ xt @ M).sum(0)
        self.t = self.t - self.learning_rate * grad_t

        grad_w = -4 * (self.W @ (xt.T @ (self.c @ delta.T @ g_out @ xt))).sum(0)
        self.W = self.W - self.learning_rate * grad_w


        print(grad_c, grad_t, grad_w)

        return delta, y_pred

    def step(self, x, y):

        loss, y_pred = self.update(x, y)

        return loss.mean(), y_pred

    def score(self, preds, targets, metric='accuracy'):
        '''
        Function: calculate the score
        '''
        ##
        #preds = self.predict(samples)  #preds: torch.Size([20000, 10]); sampels: torch.Size([20000, 3072])
        if metric=='accuracy':
            return (1.*(targets.argmax(-1) == preds.argmax(-1))).mean()*100.# targets: torch.Size([20000, 10])
        elif metric=='mse':
            return (targets - preds).pow(2).mean()

def pre_process(torchset,n_samples,num_classes=10):
    indices = list(np.random.choice(len(torchset),n_samples))

    trainset = []
    for ix in indices:
        x,y = torchset[ix]
        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        trainset.append(((x/np.linalg.norm(x)).reshape(-1),ohe_y))
    return trainset

def train():

    
    n_class = 10
    batch_size = 32
    
    transform      = transforms.Compose(
                        [transforms.ToTensor()])
    
    trainset       = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform)
    testset        = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=True, transform=transform)
    
    trainset       = pre_process(trainset, n_samples=50000, num_classes=10)
    trainloader    = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

    testset        = pre_process(testset, n_samples=50000, num_classes=10)
    testloader     = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                 shuffle=False, num_workers=2)
    
    model = udfNetWork(batch_size)
    maxsize = 6
    q = queue.Queue(maxsize)
    q.put(0.)

    epoch = 10

    for _ in range(epoch):
        for step, (train_batch_x, train_batch_y) in enumerate(trainloader):
            x_train     = train_batch_x.view(batch_size, -1)

            #print('x_train: ', x_train.shape)
            # y_train     = to_categorical(train_batch_y, n_class) # label -> one hot
            y_train       = train_batch_y
            #loss = model.step(x_train, train_batch_y)
            loss, y_pred = model.step(x_train, y_train)

            #print('y_pred', y_pred)
        
            acc = model.score(y_pred, torch.tensor(y_train))

            print('Epoch: {}, step: {}, loss: {:.4f},  acc: {}'.format(_, step, loss, acc))

            pre_loss = np.mean(list(q.queue))
            if abs(loss - pre_loss) < 1E-3:
                break
            
            if q.full():
                q.get()
            
            q.put(loss)
            #pre_loss = loss

        #y_train     = to_categorical(y_train_ori, n_class) # label -> one hot
        
    #for step, (test_batch_x, test_batch_y) in enumerate(testloader):
    #    x_test = test_batch_x.view(10000, -1)
    #    y_test = test_batch_y
    #    y_test = to_categorical(y_test, n_class) # label -> one hot
    
if __name__ == "__main__":
    train()
