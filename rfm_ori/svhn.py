import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from rfm import *

# set data path
def set_data_path():
    return "/root/ganyulu02/N-W-estimator/data/train_32x32.mat"
#     raise NotImplementedError

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

# load svhn data
transform = transforms.Compose([
    transforms.ToTensor()
])

data_path = set_data_path() ## set this data path

trainset0 = torchvision.datasets.SVHN(root='./data',
                                    split = "train",
                                    transform=transform,
                                    download=True)
testset0 = torchvision.datasets.SVHN(root='./data',
                                    split = "test",
                                    transform=transform,
                                    download=True)

trainset = pre_process(trainset0,n_samples=20000, num_classes=10)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = pre_process(testset0,n_samples=5000, num_classes=10)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

X_train, y_train = get_data(train_loader) #y_train: torch.Size([20000, 10])
X_test, y_test = get_data(test_loader)

# run rfm
rfm = LaplaceRFM(bandwidth=1.)
M, _ = rfm.fit(train_loader, test_loader, iters=1, loader=True, classif=True)