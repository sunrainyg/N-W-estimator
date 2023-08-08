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
import pdb

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

# Load MNIST dataset
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
tensor_double = tensor_float.double()
X_train = tensor_double.to(device)

# numpy_array = train_data4ma.data.reshape(-1, 3072) / 255.0
# tensor_float = torch.FloatTensor(numpy_array)
# tensor_double = tensor_float.double()
# X_train4ma = tensor_double.to(device)

# X_train = torch.cat([X_train, X_train4ma], dim=0)

# Cluster the data using KMeans with k=100
kmeans = KMeans(n_clusters=11)
kmeans.fit(X_train.cpu().numpy())

# find the cluster centers
clusters = kmeans.cluster_centers_.astype(float)
print(clusters.shape)

# make RBF network
class RBFnet(nn.Module):
    def __init__(self, clusters):
        super(RBFnet, self).__init__()
        # remember how many centers we have
        self.N = clusters.shape[0]
        # our mean and sigmas for the RBF layer
        self.sigs = nn.Parameter( torch.ones(self.N,dtype=torch.float64)*5, requires_grad=False ) # our sigmas
        self.mus  = nn.Parameter( torch.from_numpy(clusters), requires_grad=True ) # our means
        self.M    = nn.Parameter( torch.eye(3072,dtype=torch.float64, requires_grad=True))
        
        self.linear = nn.Linear(self.N, 10, dtype=torch.float64)

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
        x = x.to(torch.float64)
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
