# import required libraries
import os
import torch
import numpy as np
import torch.nn as nn
from time import time
from torch import nn, optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models, datasets, transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, f1_score
import torchvision
from sklearn.cluster import KMeans, SpectralClustering
import sys
import os
parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from N_W_estimator.N_W_estimator import KernelRegression
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

# Looking for device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform       = transforms.Compose(
                    [transforms.ToTensor()])
    
transform_ma    = transforms.Compose(
                    # [transforms.RandomHorizontalFlip(),       # 50%的概率进行水平翻转
                    # transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(50),
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor()])

# Load CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainset4ma  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_ma)
# train_dataset = ConcatDataset([train_dataset, trainset4ma])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

train_data = datasets.CIFAR10(root='./data', train=True, transform=transform)
train_data4ma = datasets.CIFAR10(root='./data', train=True, transform=transform_ma)

numpy_array = train_data.data.reshape(-1, 3072)  / 255.0
tensor_float = torch.FloatTensor(numpy_array)
X_train = tensor_float.to(device)

label = to_categorical(train_data.targets)
label = torch.FloatTensor(label)
Y_train = label.to(device)

# numpy_array = train_data4ma.data.reshape(-1, 3072) / 255.0
# tensor_float = torch.FloatTensor(numpy_array)
# tensor_double = tensor_float.double()
# X_train4ma = tensor_double.to(device)

# X_train = torch.cat([X_train, X_train4ma], dim=0)

# Cluster the data using KMeans with k=100
kmeans = KMeans(n_clusters=11)
kmeans.fit(X_train.cpu().numpy())

# find the cluster centers
clusters = kmeans.cluster_centers_.astype(np.float32)
print(clusters.shape)

# Find the M
# class estimate_M():
    
#     def __init__(self, train_M_x, train_M_y, kernel="rbf", gamma=0.4, reg=1e-3, device='cuda'):
#         self.kernel      = lambda x, z: self.laplacian_M(x, z, self.M, gamma)
#         self.gamma       = gamma
#         self.device      = device
#         self.train_M_x   = train_M_x
#         self.train_M_y   = train_M_y
#         self.reg         = reg        
    
#     def laplacian_M(self, samples, centers, M, gamma):
        
#         kernel_mat = self.euclidean_distances_M(samples, centers, M, squared=False)
#         kernel_mat.clamp_(min=0) # Guaranteed non-negative
#         gamma = 1. / gamma
#         kernel_mat.mul_(-gamma) # point-wise multiply
#         kernel_mat.exp_() #point-wise exp
#         return kernel_mat
    
#     def euclidean_distances_M(self, samples, centers, M, squared=False):
#         '''
#         Calculate the Euclidean Distances between the samples and centers, using Ma distance
#         squared = True: rbf
#         squared = False: lap
#         '''
#         ## Obtain a vector containing the square norm value for each sample point.
#         samples_norm2 = ((samples @ M) * samples).sum(-1) # torch.Size([10000])

#         if samples is centers:
#             centers_norm2 = samples_norm2
#         else:
#             centers_norm2 = ((centers @ M) * centers).sum(-1) # torch.Size([10000])
            
#         distances = -2 * (samples @ M) @ centers.T # torch.Size([10000, 10000])
#         distances.add_(samples_norm2.view(-1, 1))
#         distances.add_(centers_norm2)

#         if not squared:
#             distances.clamp_(min=0).sqrt_()
#         return distances
        
#     def update_M(self, samples, weights):

#         samples        = torch.tensor(samples).to('cuda')
        
#         K = self.kernel(samples, samples) # K_M(X, X)
#         dist = self.euclidean_distances_M(samples, samples, self.M, squared=False) # Ma distance
#         dist = torch.where(dist < 1e-10, torch.zeros(1, device=dist.device).float(), dist) # Set those small values to 0 to avoid errors caused by dividing by too small a number.

#         K = K/dist # K_M(X,X) / M_distance
#         K[K == float("Inf")] = 0. #avoid infinite big values

#         p, d = samples.shape
#         p, c = weights.shape
#         n, d = samples.shape

#         samples_term = (
#                 K # (n, p)
#                 @ weights # (p, c)
#             ).reshape(n, c, 1) ## alpha * K_M(X,X) / M_distance
                 
#         centers_term = (
#             K # (n, p)
#             @ (
#                 weights.view(p, c, 1) * (samples @ self.M).view(p, 1, d)
#             ).reshape(p, c*d) # (p, cd)
#         ).view(n, c, d) # (n, c, d) ## alpha * K_M(X,X) / M_distance

#         samples_term = samples_term * (samples @ self.M).reshape(n, 1, d) ## Mx * alpha * K_M(X,X) / M_distance

#         G = (centers_term - samples_term) / self.gamma # (n, c, d)

#         self.M = torch.einsum('ncd, ncD -> dD', G, G)/len(samples)
#         print("self.M.shape:", self.M.shape)
        
#         return self.M
    
#     def predict(self):
        
#         M = torch.eye(self.train_M_x.shape[-1])
#         self.M = M.to('cuda')
#         print("init self.M shape:", self.M.shape)
#         if self.train_M_x is not None:
#             epochs = 5
#             for epoch in range(epochs):
                
#                 alpha           = self.fit_predictor_lstsq(self.train_M_x, self.train_M_y) #alpha.shape: torch.Size([30000, 10])
#                 self.M          = self.update_M(self.train_M_x, alpha)
#                 print("One round finished")
        
#         return self.M
    
#     def fit_predictor_lstsq(self, centers, targets, batch_size=10000):
#         '''
#         Function: solve the alpha
#         Equation: alpha * K(X,X) = Y
#         Return: alpha
#         '''
#         itera           = centers.shape[0] / batch_size

#         center_batches  = torch.split(centers, batch_size, dim=0)
#         targets_batches = torch.split(targets, batch_size, dim=0)

#         alpha_batch_list = []
#         for i in range(int(itera)):
#             center_bat        = torch.tensor(center_batches[i]).to('cuda')
#             targets_bat       = torch.tensor(targets_batches[i]).to('cuda')
#             alpha_batch_i     = torch.linalg.solve(
#                                 self.kernel(center_bat, center_bat) 
#                                 + self.reg*torch.eye(len(center_bat), device=center_bat.device), 
#                                 targets_bat)
#             alpha_batch_list.append(alpha_batch_i)
#             del center_bat, targets_bat

#         alpha = torch.cat(alpha_batch_list, dim=0)
        
#         return alpha

# estimateM = estimate_M(torch.tensor(X_train), torch.tensor(Y_train))
# Ma = estimateM.predict()
######

# make RBF network
class RBFnet(nn.Module):
    def __init__(self, clusters):
        super(RBFnet, self).__init__()
        # remember how many centers we have
        self.N = clusters.shape[0]
        # our mean and sigmas for the RBF layer
        self.sigs = nn.Parameter( torch.ones(self.N,dtype=torch.float32)*5, requires_grad=False ) # our sigmas
        self.mus  = nn.Parameter( torch.from_numpy(clusters), requires_grad=True ) # our means
        self.M    = nn.Parameter( torch.eye(3072,dtype=torch.float32, requires_grad=True))
        # self.M    = nn.Parameter( Ma, requires_grad=True)
        
        self.linear = nn.Linear(self.N, 10, dtype=torch.float32)

    # def forward(self, x):
    #     diffs               = (x.unsqueeze(1) - self.mus)**2
    #     distances           = torch.sqrt((diffs.sum(dim=2)))
    #     # Calculate the Gaussian activations
    #     res                 = torch.exp((-0.5) * (distances**2) / self.sigs**2)
    #     # Set any NaN values to 0 (in case self.sigs is zero)
    #     nan_mask            = torch.isnan(res)
    #     # pdb.set_trace()
    #     res                 = torch.where(nan_mask, torch.tensor(0.0, dtype=res.dtype).to('cuda'), res)
        
    #     out                 = self.linear(res)
    #     return out
    
    def forward(self, x):
        
        ############ Ma distance ##############
        pdb.set_trace()
        samples_norm2 = ((x @ self.M) * x).sum(-1) # torch.Size([10000])
        centers_norm2 = ((self.mus @ self.M) * self.mus).sum(-1) # torch.Size([10000])
        madistances     = -2 * (x @ self.M) @ self.mus.T # torch.Size([10000, 10000])
        madistances     = madistances.add(samples_norm2.view(-1, 1))
        madistances     = madistances.add(centers_norm2)
        madistances     = madistances.clamp(min=0).sqrt()
        #######################################

        # Calculate the Gaussian activations
        res                 = torch.exp((1. / 0.6) * (madistances) / self.sigs**2)
        # Set any NaN values to 0 (in case self.sigs is zero)
        nan_mask            = torch.isnan(res)
        res                 = torch.where(nan_mask, torch.tensor(0.0, dtype=res.dtype).to('cuda'), res)

        out                 = self.linear(res)
        return out, self.M, self.mus
    
# define training function 
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 3072)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs, M_matrix, centers = model(images)
        
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss and accuracy
        running_loss += loss.item()
        y_actual = labels.data.cpu().numpy()
        _, y_pred = torch.max(outputs, dim=1)
        correct += torch.sum(y_pred == labels).item()
        total += labels.size(0)
    
    train_acc = 100 * (correct/total)
    train_loss = running_loss / len(train_loader)

    return train_acc, train_loss, M_matrix, centers

# define testing function
def test(model, test_loader, optimizer, criterion):
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
            labels_t = labels_t.to(device)

            # Forward pass
            outputs_t, _, _ = model(images_t)
            loss_t = criterion(outputs_t, labels_t)

            # Loss and accuracy
            batch_loss += loss_t.item()
            y_actual = labels_t.data.cpu().numpy()
            _, y_pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(y_pred_t == labels_t).item()
            total_t += labels_t.size(0)
            true_y.extend(y_actual)
            predicted_y.extend(y_pred_t.cpu().numpy())
        
        test_acc = (100 * correct_t / total_t)
        test_loss = (batch_loss / len(test_loader))

    return test_acc, test_loss, true_y, predicted_y


def plot_M_img(M, train_dataset, i):
    f, axarr = plt.subplots(1,2,figsize=(10, 3))
    axarr[0].axes.xaxis.set_ticklabels([])
    axarr[0].axes.yaxis.set_ticklabels([])
    axarr[1].axes.xaxis.set_ticklabels([])
    axarr[1].axes.yaxis.set_ticklabels([])

    pcm = axarr[0].imshow(torch.mean(torch.diag(M.cpu()).reshape(3,32,32),axis=0).detach().numpy(),cmap='cividis')
    axarr[0].set_title("M matrix diagonal")
    f.colorbar(mappable=pcm, ax=axarr[0], shrink=0.8,location="left")
    axarr[1].imshow(torch.moveaxis(train_dataset[6][0],0,2))
    axarr[1].set_title("Sample Image")
    plt.savefig("M_vis_{}_gd.jpg".format(i))
    plt.close()
        
model = RBFnet(clusters).to(device)

# criteria function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.8)

val_loss_list = []
val_acc_list = []
train_loss_list = []
train_acc_list = []

recall_scores = []
precision_scores = []
f1_scores = []

num_epochs = 200
for epoch in range(num_epochs):
    train_acc, train_loss, M_matrix, centers = train(model, train_loader, optimizer, criterion)
    plot_M_img(M_matrix, train_dataset, epoch)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    val_acc, val_loss, true_y, predicted_y = test(model, test_loader, optimizer, criterion)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}, Test Loss: {:.4f}, Test Acc: {:.2f}%'.format(
        epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    recall = recall_score(true_y, predicted_y, average='micro')
    precision = precision_score(true_y, predicted_y, average='micro')
    f1 = f1_score(true_y, predicted_y, average='micro')
    recall_scores.append(recall)
    precision_scores.append(precision)
    f1_scores.append(f1)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


pca = PCA(n_components=2)
centers = X_train.cpu().detach().numpy()
pca_result = pca.fit_transform(centers)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, perplexity=5, n_iter=300)
tsne_result = tsne.fit_transform(centers)

# 绘制PCA降维结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA')

# 绘制t-SNE降维结果
plt.subplot(1, 2, 2)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.title('t-SNE')
plt.savefig('cluster_centers_visualization_cifar10.png')
