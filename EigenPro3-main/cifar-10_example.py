import torch
import sys
import cifar
from eigenpro3.utils import accuracy
from eigenpro3.datasets import CustomDataset
from eigenpro3.models import KernelModel
from eigenpro3.kernels import laplacian, ntk_relu

from torchvision.datasets import CIFAR10
import os
CIFAR10('/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/data', train=True, download=True)

p = 50000 # model size
n_classes = 10

if torch.cuda.is_available():
    DEVICES = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
else:
    DEVICES = [torch.device('cpu')]

kernel_fn = lambda x, z: laplacian(x, z, bandwidth=50.0)
# kernel_fn = lambda x, z: ntk_relu(x, z, depth=2)

(X_train, y_train), (X_test, y_test), (X_train_ma, y_train_ma) = cifar.load()
X_train, y_train, X_test, y_test, X_train_ma, y_train_ma = torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test), torch.tensor(X_train_ma), torch.tensor(y_train_ma)

centers = X_train[torch.randperm(X_train.shape[0])[:p]]

## with data augmentation
#
#

testloader = torch.utils.data.DataLoader(
    CustomDataset(X_test, y_test.argmax(-1)), batch_size=512,
    shuffle=False, pin_memory=True)

# import pdb;pdb.set_trace()
# model = KernelModel(n_classes, centers, kernel_fn, X=X_train, y=y_train, devices=DEVICES)
model = KernelModel(n_classes, centers, kernel_fn, X=X_train_ma, y=y_train_ma, devices=DEVICES)
model.fit(model.train_loaders, testloader, score_fn=accuracy, epochs=200)
