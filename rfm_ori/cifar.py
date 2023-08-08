import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from rfm import *


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

def pre_process(torchset,n_samples,num_classes=10):
    indices = list(np.random.choice(len(torchset),n_samples))

    trainset = []
    for ix in indices:
        x,y = torchset[ix]
        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        trainset.append(((x/np.linalg.norm(x)).reshape(-1),ohe_y))
    return trainset

# load svhn data
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset        = torchvision.datasets.CIFAR10(root='/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/data', train=True,
                download=True, transform=transform)
testset         = torchvision.datasets.CIFAR10(root='/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/data', train=False,
                download=True, transform=transform)

trainset        = pre_process(trainset, n_samples=50000, num_classes=10)
trainloader     = torch.utils.data.DataLoader(trainset, batch_size=128,
                shuffle=False, num_workers=2)

testset         = pre_process(testset, n_samples=10000, num_classes=10)
testloader      = torch.utils.data.DataLoader(testset, batch_size=128,
                shuffle=False, num_workers=2)

X_train, y_train = get_data(trainloader) #y_train: torch.Size([20000, 10])
X_test, y_test = get_data(testloader)

# run rfm
rfm = LaplaceRFM(bandwidth=1.)
M, _ = rfm.fit(trainloader, testloader, iters=3, loader=True, classif=True)