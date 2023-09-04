## copyright Yulu Gan 2023.

import sys
sys.path.append("./")
from eigenpro import eigenpro
import numpy as np
import torchvision.transforms as transforms
import torch, torchvision
import torch.nn as nn
import pdb
import time
import kernel
import sys
import os
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
import cifar, mnist, svhn
from sklearn.metrics.pairwise import pairwise_kernels
# from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import os
# torch.backends.cuda.preferred_linalg_library("cusolver")

def to_categorical(y, num_classes=None, dtype=torch.float32):
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
    categorical[torch.arange(n), y.to(torch.long)] = 1
    output_shape = input_shape + (num_classes,)
    categorical = categorical.view(output_shape)
    return categorical


class RBFnet(nn.Module):

    def __init__(self, kernel="rbf", gamma=None, reg=1e-3, device='cuda'):
            super(RBFnet, self).__init__()
            self.kernel         = kernel
            self.gamma          = gamma
            self.device         = device
            self.pinned_list    = []
            self.reg            = reg
            self.M              = nn.Parameter( torch.eye(3072,dtype=torch.float32, requires_grad=True))
            self.kernel         = lambda x, z: self.laplacian_M(x, z, self.M, self.gamma)
            self.kernel_cpu     = lambda x, z: self.laplacian_M(x, z, self.M.cpu(), self.gamma)

    def normalize_rows(self, matrix):
        row_sums = matrix.sum(axis=1)  # 求每一行的总和
        normalized_matrix = matrix / row_sums[:, np.newaxis]  # 将每一行元素除以该行总和
        return normalized_matrix
    
    def matrix_multiplication_sum(self, weight, y_train):
        result = weight @ y_train
        return result
    
    def euclidean_distances_M(self, samples, centers, M, squared=False):
        '''
        Calculate the Euclidean Distances between the samples and centers, using Ma distance
        squared = True: rbf
        squared = False: lap
        '''
        
        ## Obtain a vector containing the square norm value for each sample point.
        # samples_norm2 = ((samples @ M) * samples).sum(-1) # torch.Size([10000])

        # if samples is centers:
        #     centers_norm2 = samples_norm2
        # else:
        #     centers_norm2 = ((centers @ M) * centers).sum(-1) # torch.Size([10000])
            
        # distances = -2 * (samples @ M) @ centers.T # torch.Size([10000, 10000])
        # distances = distances.add(samples_norm2.view(-1, 1))
        # distances = distances.add(centers_norm2)
        
        samples_norm2 = ((samples @ self.M) * samples).sum(-1) # torch.Size([10000])
        centers_norm2 = ((centers @ self.M) * centers).sum(-1) # torch.Size([10000])
        madistances     = -2 * (samples @ self.M) @ centers.T # torch.Size([10000, 10000])
        madistances     = madistances.add(samples_norm2.view(-1, 1))
        madistances     = madistances.add(centers_norm2)
        madistances     = madistances.clamp(min=0).sqrt()

        return madistances
    
    def laplacian_M(self, samples, centers, M, gamma):
        '''
        Laplacian kernel using Ma distance.

        Args:
            samples: of shape (n_sample, n_feature).
            centers: of shape (n_center, n_feature).
            M: of shape (n_feature, n_feature) or (n_feature,)
            bandwidth: kernel bandwidth. same as gamma

        Returns:
            kernel matrix of shape (n_sample, n_center).
        '''
        assert gamma > 0
        kernel_mat = self.euclidean_distances_M(samples, centers, M, squared=False)
        # kernel_mat = (samples-centers) @ self.M @ (samples-centers).T
        # kernel_mat.clamp_(min=0) # Guaranteed non-negative
        gamma = 1. / gamma
        kernel_mat = kernel_mat.mul(-gamma) # point-wise multiply
        kernel_mat = kernel_mat.exp() #point-wise exp
        # kernel_mat                 = torch.exp((1. / 0.6) * (kernel_mat))

        return kernel_mat
    
    def __del__(self):
        for pinned in self.pinned_list:
            _ = pinned.to("cpu")
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
     
    def forward(self, x_train, y_train, x_test):
        """Predict target values for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted target value.
        """

        K_test                  = self.kernel(x_train, x_test)
        normalized_K            = self.normalize_rows(K_test.T) # normalized_K.shape: (10000, 50000)
        output                  = self.matrix_multiplication_sum(normalized_K, y_train)
        
        return output, self.M


if __name__ == "__main__":
    
    # Load CIFAR10 dataset
    transform       = transforms.Compose(
                    [transforms.ToTensor()])
    train_dataset   = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset    = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=50000, shuffle=True)
    test_loader     = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    device          = "cuda"
    
val_loss_list = []
val_acc_list = []
train_loss_list = []
train_acc_list = []


model = RBFnet(gamma=0.4).to(device)
# criteria function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=0.8)


num_epochs = 100
for epoch in range(num_epochs):
    
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 3072)
        images = images.to(device)
        labels = to_categorical(labels)
        labels = labels.to(device).to(torch.float32)

        # Forward pass
        outputs, M_matrix = model(images, labels, images)
        # pdb.set_trace()
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss and accuracy
        running_loss += loss.item()
        _, y_pred = torch.max(outputs, dim=1)
        _, y_gt   = torch.max(labels, dim=1)
        correct += torch.sum(y_pred == y_gt).item()
        total += labels.size(0)
        
        train_acc = 100 * (correct/total)
        train_loss = running_loss / len(train_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

    model.eval()   
    true_y = []
    predicted_y = []

    batch_loss = 0
    total_t = 0
    correct_t = 0

    with torch.no_grad():
        for i, (images_t, labels_t) in enumerate(test_loader):
            images_t = images_t.view(-1, 3072)
            images_t = images_t.to(device)
            labels_t = to_categorical(labels_t)
            labels_t = labels_t.to(device).to(torch.float32)

            # Forward pass
            outputs_t, _ = model(images, labels, images_t)
            loss_t = criterion(outputs_t, labels_t)

            # Loss and accuracy
            batch_loss += loss_t.item()
            y_actual = labels_t.data.cpu().numpy()
            _, y_pred_t = torch.max(outputs_t, dim=1)
            _, y_gt_t   = torch.max(labels_t, dim=1)
            correct_t += torch.sum(y_pred_t == y_gt_t).item()
            total_t += labels_t.size(0)
            true_y.extend(y_actual)
            predicted_y.extend(y_pred_t.cpu().numpy())
        
        test_acc = (100 * correct_t / total_t)
        test_loss = (batch_loss / len(test_loader))
        val_acc_list.append(test_acc)
        val_loss_list.append(test_loss)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}, Test Loss: {:.4f}, Test Acc: {:.2f}%'.format(
            epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc))