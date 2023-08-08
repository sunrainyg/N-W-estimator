## copyright Yulu Gan 2023.

import sys
sys.path.append("./")
from eigenpro import eigenpro
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pdb
import time
import kernel
import cifar, mnist, svhn
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import os
# torch.backends.cuda.preferred_linalg_library("cusolver")



class KernelRegression(BaseEstimator, RegressorMixin):
    """Nadaraya-Watson kernel regression with automatic bandwidth selection.

    This implements Nadaraya-Watson kernel regression with (optional) automatic
    bandwith selection of the kernel via leave-one-out cross-validation. Kernel
    regression is a simple non-parametric kernelized technique for learning
    a non-linear relationship between input variable(s) and a target variable.

    Parameters
    ----------
    kernel : string or callable, default="rbf"
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.

    gamma : float, default=None
        Gamma parameter for the RBF ("bandwidth"), polynomial,
        exponential chi2 and sigmoid kernels. Interpretation of the default
        value is left to the kernel; see the documentation for
        sklearn.metrics.pairwise. Ignored by other kernels. If a sequence of
        values is given, one of these values is selected which minimizes
        the mean-squared-error of leave-one-out cross-validation.

    See also
    --------

    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.
    """

    def __init__(self, block=1, kernel="rbf", gamma=None, reg=1e-3, device='cuda'):
            self.kernel      = kernel
            self.gamma       = gamma
            self.block       = block
            self.device      = device
            self.pinned_list = []
            self.reg         = reg

    def normalize_rows(self, matrix):
        row_sums = matrix.sum(axis=1)  # 求每一行的总和
        normalized_matrix = matrix / row_sums[:, np.newaxis]  # 将每一行元素除以该行总和
        return normalized_matrix

    def fit(self, x_train, y_train, train_M_x=None, train_M_y=None, M=None):
        """Fit the model

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values

        Returns
        -------
        self : object
            Returns self.
        """
        
        M = torch.eye(x_train.shape[-1])
        self.M = M.to('cuda')
        print("init self.M shape:", self.M.shape)
            
            
        self.kernel         = lambda x, z: self.laplacian_M(x, z, self.M, self.gamma)
        self.kernel_cpu     = lambda x, z: self.laplacian_M(x, z, self.M.cpu(), self.gamma)
        self.x_train        = x_train
        self.y_train        = y_train
        self.train_M_x      = train_M_x
        self.train_M_y      = train_M_y

        # if hasattr(self.gamma, "__iter__"):
        #     self.gamma = self._optimize_gamma(self.gamma)

        return self
    
    def matrix_multiplication_sum(self, weight, y_train):

        # assert weight.shape[1] == y_train.shape[0], "矩阵 A 的列数必须等于矩阵 B 的行数"
        # result = np.zeros((weight.shape[0], y_train.shape[1]))

        # row_sum = 0
        # for i in range(weight.shape[0]):
        #     for j in range(weight.shape[1]):
        #         logit = weight[i][j] * y_train[j]
        #         row_sum = row_sum + logit 
        #     result[i] = row_sum
        #     row_sum = 0
        result = weight @ y_train
        return result
    
    def euclidean_distances_M(self, samples, centers, M, squared=False):
        '''
        Calculate the Euclidean Distances between the samples and centers, using Ma distance
        squared = True: rbf
        squared = False: lap
        '''
        
        ## Obtain a vector containing the square norm value for each sample point.
        samples_norm2 = ((samples @ M) * samples).sum(-1) # torch.Size([10000])

        if samples is centers:
            centers_norm2 = samples_norm2
        else:
            centers_norm2 = ((centers @ M) * centers).sum(-1) # torch.Size([10000])
            
        distances = -2 * (samples @ M) @ centers.T # torch.Size([10000, 10000])
        distances.add_(samples_norm2.view(-1, 1))
        distances.add_(centers_norm2)

        if not squared:
            distances.clamp_(min=0).sqrt_()
        return distances

    def euclidean_distances(self, samples, centers, squared=False):
    
        samples_norm2 = np.sum(samples**2, axis=-1)
        
        if samples is centers:
            centers_norm2 = samples_norm2
        else:
            centers_norm2 = np.sum(centers**2, axis=-1)
        
        distances = -2 * np.dot(samples, centers.T)
        distances += samples_norm2[:, np.newaxis]
        distances += centers_norm2
        
        if not squared:
            np.clip(distances, 0, None, out=distances)
            np.sqrt(distances, out=distances)
        
        return distances
    
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
        kernel_mat.clamp_(min=0) # Guaranteed non-negative
        gamma = 1. / gamma
        kernel_mat.mul_(-gamma) # point-wise multiply
        kernel_mat.exp_() #point-wise exp
        return kernel_mat
 
    def laplacian(self, samples, centers, bandwidth):
        '''Laplacian kernel.

        Args:
            samples: of shape (n_sample, n_feature).
            centers: of shape (n_center, n_feature).
            bandwidth: kernel bandwidth.

        Returns:
            kernel matrix of shape (n_sample, n_center).
        '''
        assert bandwidth > 0
        kernel_mat = self.euclidean_distances(samples, centers, squared=False)
        kernel_mat = np.clip(kernel_mat, 0, None)
        gamma = 1.0 / bandwidth
        kernel_mat *= -gamma
        np.exp(kernel_mat, out=kernel_mat)
        return kernel_mat

    def tensor(self, data, dtype=None, release=False):
        tensor = torch.as_tensor(data, dtype=dtype, device=self.device)
        if release:
            self.pinned_list.append(tensor)
        return tensor
    
    def __del__(self):
        for pinned in self.pinned_list:
            _ = pinned.to("cpu")
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
    
    def update_M(self, samples, weights):
        '''
        Input:  samples - X_train; same as centers
                self.weights - alpha
                
        Notion: K := K_M(X,X) / ||x-z||_M
                samples_term := Mx * K_M(x,z)
                centers_term := Mz * K_M(x,z)
                
        Equation: grad(K_M(x,z)) = (Mx-Mz) * (K_M(x,z)) / (L||x-z||_M),
        where x is samples, z is center,
        
        '''
        samples        = torch.tensor(samples).to('cuda')
        
        K = self.kernel(samples, samples) # K_M(X, X)
        dist = self.euclidean_distances_M(samples, samples, self.M, squared=False) # Ma distance
        dist = torch.where(dist < 1e-10, torch.zeros(1, device=dist.device).float(), dist) # Set those small values to 0 to avoid errors caused by dividing by too small a number.

        K = K/dist # K_M(X,X) / M_distance
        K[K == float("Inf")] = 0. #avoid infinite big values

        p, d = samples.shape
        p, c = weights.shape
        n, d = samples.shape

        samples_term = (
                K # (n, p)
                @ weights # (p, c)
            ).reshape(n, c, 1) ## alpha * K_M(X,X) / M_distance
                 
        centers_term = (
            K # (n, p)
            @ (
                weights.view(p, c, 1) * (samples @ self.M).view(p, 1, d)
            ).reshape(p, c*d) # (p, cd)
        ).view(n, c, d) # (n, c, d) ## alpha * K_M(X,X) / M_distance

        samples_term = samples_term * (samples @ self.M).reshape(n, 1, d) ## Mx * alpha * K_M(X,X) / M_distance

        G = (centers_term - samples_term) / self.gamma # (n, c, d)

        self.M = torch.einsum('ncd, ncD -> dD', G, G)/len(samples)
        print("self.M.shape:", self.M.shape)
        
        return self.M
    
    def fit_predictor_lstsq(self, centers, targets, batch_size=10000):
        '''
        Function: solve the alpha
        Equation: alpha * K(X,X) = Y
        Return: alpha
        '''
        itera           = centers.shape[0] / batch_size

        # 使用torch.chunk函数将data分成5份，每份大小为10000
        center_batches  = torch.split(centers, batch_size, dim=0)
        targets_batches = torch.split(targets, batch_size, dim=0)

        # center_batches是一个包含五个张量的列表，每个张量的形状为（10000，3072）
        # data_batches[0]包含前10000行，data_batches[1]包含接下来的10000行，以此类推
        alpha_batch_list = []
        for i in range(int(itera)):
            center_bat        = torch.tensor(center_batches[i]).to('cuda')
            targets_bat       = torch.tensor(targets_batches[i]).to('cuda')
            alpha_batch_i     = torch.linalg.solve(
                                self.kernel(center_bat, center_bat) 
                                + self.reg*torch.eye(len(center_bat), device=center_bat.device), 
                                targets_bat)
            alpha_batch_list.append(alpha_batch_i)
            del center_bat, targets_bat

        alpha = torch.cat(alpha_batch_list, dim=0)
        
        return alpha

    def fit_predictor_eigenpro(self, x_train, y_train, x_test, y_test):
        n_class = 10

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        model = eigenpro.FKR_EigenPro(self.kernel_tensor, x_train, n_class, device=device)

        res = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 2, 5], mem_gb=12)
        return model.weight
       
    def predict(self, x_test, y_test):
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
        # K_test          = pairwise_kernels(self.x_train, x_test, metric=self.kernel, gamma=self.gamma) # K.shape (49000, 10000)

        # if self.train_M_x is not None:
        #     epochs = 5
        #     batch_size = 10000
        #     for epoch in range(epochs):
                
        #         alpha           = self.fit_predictor_lstsq(self.train_M_x, self.train_M_y) #alpha.shape: torch.Size([30000, 10])
        #         self.M          = self.update_M(self.train_M_x, alpha)
        #         print("One round finished")

        K_test          = self.kernel_cpu(self.x_train, x_test)
        normalized_K    = self.normalize_rows(K_test.T) # normalized_K.shape: (10000, 50000)
        output          = self.matrix_multiplication_sum(normalized_K, self.y_train)
        
        eval_metrics = {}
        
        y_class = torch.argmax(y_test, dim=-1)
        p_class = torch.argmax(output, dim=-1)
        eval_metrics = torch.sum(y_class == p_class).item() / len(y_class)
        print(eval_metrics)
        
        return output
    
    def forward(self, x_test, y_test):
        # block = self.block
        output = self.predict(x_test, y_test)
        return output, self.M

if __name__ == "__main__":
    
    def plot_M_img(M):
        f, axarr = plt.subplots(1,2,figsize=(10, 3))
        axarr[0].axes.xaxis.set_ticklabels([])
        axarr[0].axes.yaxis.set_ticklabels([])
        axarr[1].axes.xaxis.set_ticklabels([])
        axarr[1].axes.yaxis.set_ticklabels([])

        pcm = axarr[0].imshow(torch.mean(torch.diag(M.cpu()).reshape(3,32,32),axis=0),cmap='cividis')
        axarr[0].set_title("M matrix diagonal")
        f.colorbar(mappable=pcm, ax=axarr[0], shrink=0.8,location="left")
        axarr[1].imshow(torch.moveaxis(trainset[18][0],0,2))
        axarr[1].set_title("Sample Image")
        plt.savefig("M_vis.jpg")
        plt.close()
    
    def plot_M_img_mnisit(M):
        f, axarr = plt.subplots(1,2,figsize=(10, 3))
        axarr[0].axes.xaxis.set_ticklabels([])
        axarr[0].axes.yaxis.set_ticklabels([])
        axarr[1].axes.xaxis.set_ticklabels([])
        axarr[1].axes.yaxis.set_ticklabels([])

        pcm = axarr[0].imshow(torch.mean(torch.diag(M.cpu()).reshape(28,28).unsqueeze(0).expand(3, -1, -1),axis=0).detach().numpy(),cmap='cividis')
        axarr[0].set_title("M matrix diagonal")
        f.colorbar(mappable=pcm, ax=axarr[0], shrink=0.8,location="left")
        axarr[1].imshow(torch.moveaxis(trainset[3][0],0,2))
        axarr[1].set_title("Sample Image")
        plt.savefig("M_vis_mnist.jpg")
        plt.close()
    
    ##### cifar
    # n_class = 10
    # cifar10_dir = './data/cifar-10-batches-py'
    # (x_train, y_train), (x_test, y_test), (x_train4ma, y_train4ma), trainset = cifar.load_10classes(cifar10_dir)
    # # (x_train, y_train), (x_test, y_test), (x_train4ma, y_train4ma) = mnist.load_10classes(cifar10_dir)
    
    # x_train         = torch.tensor(x_train).to("cpu")
    # y_train         = torch.tensor(y_train).to("cpu")
    # x_test          = torch.tensor(x_test).to("cpu")
    # y_test          = torch.tensor(y_test).to("cpu")
    # x_train4ma      = torch.tensor(x_train4ma).to('cpu')
    # y_train4ma      = torch.tensor(y_train4ma).to('cpu')
            
    # x_train         = x_train.detach().to(torch.float32)
    # y_train         = y_train.detach().to(torch.float32)
    # x_test          = x_test.detach().to(torch.float32)
    # y_test          = y_test.detach().to(torch.float32)
    # x_train4ma      = x_train4ma.detach().to(torch.float32)
    # y_train4ma      = y_train4ma.detach().to(torch.float32)
    # # Fit regression models
    
    # x_train_part1   = x_train[0:30000]
    # x_train_part2   = x_train4ma[30000:50000]
    # x_train_part3   = x_train[20000:30000]
    # x_train_part4   = x_train[30000:40000]
    # x_train_part5   = x_train[40000:50000]
    
    # y_train_part1   = y_train[0:30000]
    # y_train_part2   = y_train4ma[30000:50000]
    # y_train_part3   = y_train[20000:30000]
    # y_train_part4   = y_train[30000:40000]
    # y_train_part5   = y_train[40000:50000]
    
    # x_train4ma_part = x_train4ma[0:60000]
    # y_train4ma_part = y_train4ma[0:60000]
    
    # t0 = time.time()
    # kr = KernelRegression(kernel="rbf", gamma=0.4)
    # # #### 1 layer:
    # y_kr, M = kr.fit(x_train_part1, y_train_part1, x_train4ma_part, y_train4ma_part).forward(x_test, y_test) # 1 layer
    # plot_M_img(M)
    #### 2 layer:
    ## train
    # expectation, Maha      = kr.fit(x_train_part1, y_train_part1, x_train4ma_part, y_train4ma_part).forward(x_train_part2, y_train_part2)
    # error_gt         = y_train_part2 - expectation #每一层存x, y
    
    # # ## inference
    # expectation_inf, Maha  = kr.fit(x_train_part1, y_train_part1, x_train4ma_part, y_train4ma_part).forward(x_test, y_test)
    # error_est, Maha      = kr.fit(x_train_part2, error_gt, x_train4ma_part, y_train4ma_part).forward(x_test, y_test)
    # result           = expectation_inf + error_est
    
    # eval_metrics = {}
    # y_class =  torch.argmax(y_test, dim=-1)
    # p_class =  torch.argmax(result, dim=-1)
    # eval_metrics['multiclass-acc'] = torch.sum(y_class == p_class).item() / len(y_class)
    # print(eval_metrics)
    
    # #### 3 layer:
    # ## train
    # expectation      = kr.fit(x_train_part1, y_train_part1).forward(x_train_part2, y_train_part2)
    # error_gt2        = y_train_part2 - expectation #每一层存x, y
    # error_est2       = kr.fit(x_train_part2, error_gt2).forward(x_train_part3, y_train_part3)
    # error_gt3        = y_train_part3 - error_est2 - expectation
    
    # ## inference
    # expectation_inf  = kr.fit(x_train_part1, y_train_part1).forward(x_test, y_test) #layer1
    # error_est1       = kr.fit(x_train_part2, error_gt2).forward(x_test, y_test) #layer2
    # error_est2       = kr.fit(x_train_part3, error_gt3).forward(x_test, y_test) #layer3
    # result           = expectation_inf + error_est1 + error_est2
    
    # eval_metrics = {}
    # y_class = np.argmax(y_test, axis=-1)
    # p_class = np.argmax(result, axis=-1)
    # eval_metrics['multiclass-acc'] = np.mean(y_class == p_class)
    # print(eval_metrics)
    
    #### 5 layer:
    
    # expectation, _      = kr.fit(x_train_part1, y_train_part1, x_train4ma_part, y_train4ma_part).forward(x_train_part2, y_train_part2)
    # error_gt2        = y_train_part2 - expectation #每一层存x, y
    
    # error_est2, _        = kr.fit(x_train_part2, error_gt2, x_train4ma_part, y_train4ma_part).forward(x_train_part3, y_train_part3)
    # error_gt3        = y_train_part3 - error_est2 - expectation
    
    # error_est3, _        = kr.fit(x_train_part3, error_gt3, x_train4ma_part, y_train4ma_part).forward(x_train_part4, y_train_part4)
    # error_gt4        = y_train_part4 - error_est3 - error_est2 - expectation
    
    # error_est4, _        = kr.fit(x_train_part4, error_gt4, x_train4ma_part, y_train4ma_part).forward(x_train_part5, y_train_part5)
    # error_gt5        = y_train_part5 - error_est4 - error_est3 - error_est2 - expectation
    
    
    # ## inference
    # expectation_inf, _   = kr.fit(x_train_part1, y_train_part1, x_train4ma_part, y_train4ma_part).forward(x_test, y_test) #layer1
    # error_est1, _        = kr.fit(x_train_part2, error_gt2, x_train4ma_part, y_train4ma_part).forward(x_test, y_test) #layer2
    # error_est2, _        = kr.fit(x_train_part3, error_gt3, x_train4ma_part, y_train4ma_part).forward(x_test, y_test) #layer3
    # error_est3, _        = kr.fit(x_train_part4, error_gt4, x_train4ma_part, y_train4ma_part).forward(x_test, y_test) #layer4
    # error_est4, _        = kr.fit(x_train_part5, error_gt5, x_train4ma_part, y_train4ma_part).forward(x_test, y_test) #layer4
    
    # result5           = expectation_inf + error_est1 + error_est2 + error_est3 + error_est4
    # result4           = expectation_inf + error_est1 + error_est2 + error_est3
    # result3           = expectation_inf + error_est1 + error_est2
    # result2           = expectation_inf + error_est1
    # result1           = expectation_inf
    
    
    # eval_metrics = {}
    # y_class =  torch.argmax(y_test, dim=-1)
    # p_class1 = torch.argmax(result1, dim=-1)
    # p_class2 = torch.argmax(result2, dim=-1)
    # p_class3 = torch.argmax(result3, dim=-1)
    # p_class4 = torch.argmax(result4, dim=-1)
    # p_class5 = torch.argmax(result5, dim=-1)
    
    # eval_metrics['multiclass-acc of 1st layer'] = torch.sum(y_class == p_class1).item() / len(y_class)
    # eval_metrics['multiclass-acc of 2nd layer'] = torch.sum(y_class == p_class2).item() / len(y_class)
    # eval_metrics['multiclass-acc of 3rd layer'] = torch.sum(y_class == p_class3).item() / len(y_class)
    # eval_metrics['multiclass-acc of 4th layer'] = torch.sum(y_class == p_class4).item() / len(y_class)
    # eval_metrics['multiclass-acc of 5th layer'] = torch.sum(y_class == p_class5).item() / len(y_class)
    # print(eval_metrics)
    
    
    
    # print("KR including bandwith fitted in %.3f s" \
    #     % (time.time() - t0))
    
    # ### minist
    n_class = 10
    
    (x_train, y_train), (x_test, y_test), trainset = mnist.load_10classes()
    x_train         = torch.tensor(x_train).to("cpu")
    y_train         = torch.tensor(y_train).to("cpu")
    x_test          = torch.tensor(x_test).to("cpu")
    y_test          = torch.tensor(y_test).to("cpu")
            
    x_train         = x_train.detach().to(torch.float32)
    y_train         = y_train.detach().to(torch.float32)
    x_test          = x_test.detach().to(torch.float32)
    y_test          = y_test.detach().to(torch.float32)
    
    x_train_part1 = x_train[0:20000]
    x_train_part2 = x_train[20000:40000]
    
    y_train_part1 = y_train[0:20000]
    y_train_part2 = y_train[20000:40000]
    
    # Fit regression models
    kr = KernelRegression(kernel="laplacian", gamma=0.5)
    expectation,M      = kr.fit(x_train_part1, y_train_part1, x_train, y_train).forward(x_train_part2, y_train_part2)
    plot_M_img_mnisit(M)
    error_gt         = y_train_part2 - expectation #每一层存x, y
    
    # ## inference
    expectation_inf,_   = kr.fit(x_train_part1, y_train_part1, x_train, y_train).forward(x_test, y_test)
    error_est,_         = kr.fit(x_train_part2, error_gt, x_train, y_train).forward(x_test, y_test)
    result           = expectation_inf + error_est
    
    eval_metrics = {}
    y_class =  torch.argmax(y_test, dim=-1)
    p_class =  torch.argmax(result, dim=-1)
    eval_metrics['multiclass-acc'] = torch.sum(y_class == p_class).item() / len(y_class)
    print(eval_metrics)
    
    t0 = time.time()
    kr = KernelRegression(kernel="laplacian", gamma=0.5)
    y_kr = kr.fit(x_train, y_train).forward(x_test, y_test) # X.shape: (100, 1), y.shape: (100,) np.expand_dims(y, axis=1).shape
    print("KR including bandwith fitted in %.3f s" \
        % (time.time() - t0))


    # #### svhn
    # n_class = 10
    # (x_train, y_train), (x_test, y_test) = svhn.load_10classes()
    # kr = KernelRegression(kernel="rbf", gamma=0.3)
    
    # x_train_part1 = x_train[0:10000]
    # x_train_part2 = x_train[10000:20000]
    
    # y_train_part1 = y_train[0:10000]
    # y_train_part2 = y_train[10000:20000]
    # #### 2 layer:
    # ## train
    # expectation,_      = kr.fit(x_train_part1, y_train_part1, x_train, y_train).forward(x_train_part2, y_train_part2)
    # error_gt         = y_train_part2 - expectation #每一层存x, y
    
    # # ## inference
    # expectation_inf,_   = kr.fit(x_train_part1, y_train_part1, x_train, y_train).forward(x_test, y_test)
    # error_est,_         = kr.fit(x_train_part2, error_gt, x_train, y_train).forward(x_test, y_test)
    # result           = expectation_inf + error_est
    
    # eval_metrics = {}
    # y_class =  torch.argmax(y_test, dim=-1)
    # p_class =  torch.argmax(result, dim=-1)
    # eval_metrics['multiclass-acc'] = torch.sum(y_class == p_class).item() / len(y_class)
    # print(eval_metrics)