# Copyright Yulu Gan 2023.


import cifar
import torch
import pdb
import eigenpro

n_class = 10
cifar10_dir = './data/cifar-10-batches-py'
(x_train, y_train), (x_test, y_test), (x_train_ma, y_train_ma) = cifar.load()
# x_train, y_train, x_test, y_test = x_train.astype('float32'), \
#     y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')
# x_train (49000, 3072); y_train (49000, 10); x_test (10000, 3072); y_test (10000, 10)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = eigenpro.FKR_EigenPro(x_train, y_train, n_class, device=device)
res, acc = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 2, 5, 10], mem_gb=12)
print("acc", acc)

## optimize the bandwidth
# best_acc = 0.0
# best_gamma = 1
# GAMMA_LIST = [0.1 * (2.0 ** i)for i in range(-5, 5)]
# for gamma_layer1 in GAMMA_LIST:
#     print("gamma:", gamma_layer1)
#     kernel_fn = lambda x,y: kernel.laplacian(x, y, bandwidth=gamma_layer1)
#     model = eigenpro.FKR_EigenPro(kernel_fn, x_train, n_class, device=device)
#     res, acc = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 2, 5], mem_gb=12)
#     if acc > best_acc:
#         best_acc = acc
#         best_gamma = gamma_layer1
#         print ("best acc:", best_acc, "\tbest gamma:", best_gamma)