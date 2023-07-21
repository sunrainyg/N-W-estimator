## copyright Yulu Gan 2023.

import sys
sys.path.append("./")
from eigenpro import eigenpro
import numpy as np
import torch
import pdb
import time
import kernel
import cifar
import mnist
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin


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

    def __init__(self, block=1, kernel="rbf", gamma=None):
            self.kernel = kernel
            self.gamma  = gamma
            self.block  = block
            self.device = [torch.device('cpu')]

    def normalize_rows(self, matrix):
        row_sums = matrix.sum(axis=1)  # 求每一行的总和
        normalized_matrix = matrix / row_sums[:, np.newaxis]  # 将每一行元素除以该行总和
        return normalized_matrix

    def fit(self, x_train, y_train, M=None):
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
        if M is None:
            M = np.eye(x_train.shape[-1])
            self.M = M
        self.kernel         = lambda x, z: self.laplacian_M(x, z, self.M, self.gamma) 
        self.kernel_tensor  = lambda x, z: self.laplacian_M_tensor(x, z, self.M, self.gamma) 
        self.x_train = x_train
        self.y_train = y_train

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
        result = np.dot(weight, y_train)
        return result
    
    def euclidean_distances_M(self, samples, centers, M, squared=True):
        '''
        Calculate the Euclidean Distances between the samples and centers, using Ma distance
        '''

        ## Obtain a vector containing the square norm value for each sample point.
        samples_norm2 = np.sum((samples @ M) * samples, axis=-1)

        if samples is centers:
            centers_norm2 = samples_norm2
        else:
            centers_norm2 = np.sum((centers @ M) * centers, axis=-1)

        distances = -2 * np.dot(samples @ M, centers.T)
        distances += samples_norm2[:, np.newaxis]
        distances += centers_norm2

        if not squared:
            np.clip(distances, 0, None, out=distances)
            np.sqrt(distances, out=distances)

        return distances
    
    def euclidean_distances_M_tensor(self, samples, centers, M, squared=True):
        '''
        Calculate the Euclidean Distances between the samples and centers, using Ma distance
        '''
        
        ## Obtain a vector containing the square norm value for each sample point.
        samples_norm2 = ((samples @ M) * samples).sum(-1)

        if samples is centers:
            centers_norm2 = samples_norm2
        else:
            centers_norm2 = ((centers @ M) * centers).sum(-1)
            
        samples = samples.double()
        centers = centers.double()
        M       = M.astype(np.float64)
        
        distances = -2 * (samples @ M) @ centers.T
        distances.add_(samples_norm2.view(-1, 1))
        distances.add_(centers_norm2)

        if not squared:
            distances.clamp_(min=0).sqrt_()

        return distances
    
    def euclidean_distances(self, samples, centers, squared=True):
    
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
    
    def laplacian_M(self, samples, centers, M, bandwidth):
        '''
        Laplacian kernel using Ma distance.

        Args:
            samples: of shape (n_sample, n_feature).
            centers: of shape (n_center, n_feature).
            M: of shape (n_feature, n_feature) or (n_feature,)
            bandwidth: kernel bandwidth.

        Returns:
            kernel matrix of shape (n_sample, n_center).
        '''
        assert bandwidth > 0
        kernel_mat = self.euclidean_distances_M(samples, centers, M, squared=False)
        kernel_mat = np.clip(kernel_mat, 0, None)
        gamma = 1.0 / bandwidth
        kernel_mat *= -gamma
        np.exp(kernel_mat, out=kernel_mat)
        return kernel_mat
    
    def laplacian_M_tensor(self, samples, centers, M, bandwidth):
        '''
        Equation: exp(\gamma * (x-xi)M(x-xi)^T)
        '''
        
        assert bandwidth > 0
        kernel_mat = self.euclidean_distances_M_tensor(samples, centers, M, squared=False)
        kernel_mat.clamp_(min=0) # Guaranteed non-negative
        gamma = 1. / bandwidth
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

    def update_M(self, samples, centers, weights):
        '''
        Input:  samples - test set
                centers - training set 
                weights - alpha
                
        Notion: K := K_M(X,X) / ||x-z||_M
                samples_term := Mx * K_M(x,z)
                centers_term := Mz * K_M(x,z)
                
        Equation: grad(K_M(x,z)) = (Mx-Mz) * (K_M(x,z)) / (L||x-z||_M),
        where x is samples, z is center,
        '''

        K = self.kernel(samples, centers)  # K_M(X, X)

        dist = np.linalg.norm(samples[:, np.newaxis] - centers, axis=-1)  # Ma distance using numpy's norm
        dist = np.where(dist < 1e-10, np.zeros(1, dtype=dist.dtype), dist)  # Set those small values to 0 to avoid errors caused by dividing by too small a number.

        K = K / dist  # K_M(X,X) / M_distance
        K[np.isinf(K)] = 0.  # avoid infinite big values

        p, d = centers.shape
        p, c = self.weights.shape
        n, d = samples.shape

        samples_term = (
                K  # (n, p)
                @ self.weights  # (p, c)
            ).reshape(n, c, 1)  ## alpha * K_M(X,X) / M_distance

        if self.diag:
            centers_term = (
                K  # (n, p)
                @ (
                    self.weights.reshape(p, c, 1) * (centers * self.M).reshape(p, 1, d)
                ).reshape(p, c * d)  # (p, cd)
            ).reshape(n, c, d)  # (n, c, d)

            samples_term = samples_term * (samples * self.M).reshape(n, 1, d)
        else:
            centers_term = (
                K  # (n, p)
                @ (
                    self.weights.reshape(p, c, 1) * (centers @ self.M).reshape(p, 1, d)
                ).reshape(p, c * d)  # (p, cd)
            ).reshape(n, c, d)  # (n, c, d)

            samples_term = samples_term * (samples @ self.M).reshape(n, 1, d)

        G = (centers_term - samples_term) / self.gamma  # (n, c, d)

        if self.centering:
            G = G - G.mean(0)  # (n, c, d)

        if self.diag:
            np.einsum('ncd, ncd -> d', G, G) / len(samples)
        else:
            self.M = np.einsum('ncd, ncD -> dD', G, G) / len(samples)
            print("self.M.shape:", self.M.shape)

    def fit_predictor_lstsq(self, centers, targets):
        '''
        Function: solve the alpha
        Equation: alpha * K(X,X) = Y
        Return: alpha
        '''
        
        K = self.kernel(centers, centers)
        reg_matrix = self.reg * np.eye(len(centers))
        alpha = np.linalg.solve(K + reg_matrix, targets)
        
        return alpha
    
    def fit_predictor_eigenpro(self, x_train, y_train, x_test, y_test):
        n_class = 10
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = eigenpro.FKR_EigenPro(self.kernel_tensor, x_train, n_class, device=device)

        x_train, y_train, x_test, y_test = x_train.astype('float64'), \
            y_train.astype('float64'), x_test.astype('float64'), y_test.astype('float64')
        res = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 2, 5], mem_gb=12)
        pdb.set_trace()
        return self.model.weigts
       
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
        
        # epochs = 5
        # for epoch in range(epochs):
        #     alpha       = self.fit_predictor_lstsq(x_train, y_train)
        #     self.M      = self.update_M(x_test, x_train, alpha)
        #     print("First round finished, M:", self.M)
        ################
        alpha           = self.fit_predictor_eigenpro(x_train, y_train, x_test, y_test)
        self.M          = self.update_M(x_test, x_train, alpha)
        ################
        
        K_test          = self.kernel(self.x_train, x_test)
        normalized_K    = self.normalize_rows(K_test.T) # normalized_K.shape: (10000, 50000)
        output          = self.matrix_multiplication_sum(normalized_K, self.y_train)

        # output = np.matmul(normalized_K, self.y_train)
        
        eval_metrics = {}
        y_class = np.argmax(y_test, axis=-1)
        p_class = np.argmax(output, axis=-1)
        eval_metrics['multiclass-acc'] = np.mean(y_class == p_class)
        print(eval_metrics)
        
        return output
    
    def forward(self, x_test, y_test):
        # block = self.block
        output = self.predict(x_test, y_test)
        
        return output
    

if __name__ == "__main__":
    
    ##### cifar
    n_class = 10
    cifar10_dir = '/Users/yulu/N-W-estimator/dataset/cifar10/cifar-10-batches-py'
    (x_train, y_train), (x_test, y_test) = cifar.load_10classes(cifar10_dir)
    x_train, y_train, x_test, y_test = x_train.astype('float32'), \
    y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')
    # Fit regression models
    
    x_train_part1 = x_train[0:10000]
    x_train_part2 = x_train[10000:20000]
    x_train_part3 = x_train[20000:30000]
    x_train_part4 = x_train[30000:40000]
    x_train_part5 = x_train[40000:50000]
    
    y_train_part1 = y_train[0:10000]
    y_train_part2 = y_train[10000:20000]
    y_train_part3 = y_train[20000:30000]
    y_train_part4 = y_train[30000:40000]
    y_train_part5 = y_train[40000:50000]
    
    t0 = time.time()
    kr = KernelRegression(kernel="rbf", gamma=0.8)
    # y_kr = kr.fit(X, y).forward(X) # X.shape: (100, 1), y.shape: (100,) np.expand_dims(y, axis=1).shape
    
    #### 1 layer:
    # y_kr = kr.fit(x_train_part1, y_train_part1).forward(x_test, y_test) # 1 layer
    
    # #### 2 layer:
    # ## train
    # expectation      = kr.fit(x_train_part1, y_train_part1).forward(x_train_part2, y_train_part2)
    # error_gt         = y_train_part2 - expectation #每一层存x, y
    
    # ## inference
    # expectation_inf  = kr.fit(x_train_part1, y_train_part1).forward(x_test, y_test)
    # error_est        = kr.fit(x_train_part2, error_gt).forward(x_test, y_test)
    # result           = expectation_inf + error_est
    
    # eval_metrics = {}
    # y_class = np.argmax(y_test, axis=-1)
    # p_class = np.argmax(result, axis=-1)
    # eval_metrics['multiclass-acc'] = np.mean(y_class == p_class)
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
    ## train
    expectation      = kr.fit(x_train_part1, y_train_part1).forward(x_train_part2, y_train_part2)
    error_gt2        = y_train_part2 - expectation #每一层存x, y
    
    error_est2       = kr.fit(x_train_part2, error_gt2).forward(x_train_part3, y_train_part3)
    error_gt3        = y_train_part3 - error_est2 - expectation
    
    error_est3       = kr.fit(x_train_part3, error_gt3).forward(x_train_part4, y_train_part4)
    error_gt4        = y_train_part4 - error_est3 - error_est2 - expectation
    
    error_est4       = kr.fit(x_train_part4, error_gt4).forward(x_train_part5, y_train_part5)
    error_gt5        = y_train_part5 - error_est4 - error_est3 - error_est2 - expectation
    
    
    ## inference
    expectation_inf  = kr.fit(x_train_part1, y_train_part1).forward(x_test, y_test) #layer1
    error_est1       = kr.fit(x_train_part2, error_gt2).forward(x_test, y_test) #layer2
    error_est2       = kr.fit(x_train_part3, error_gt3).forward(x_test, y_test) #layer3
    error_est3       = kr.fit(x_train_part4, error_gt4).forward(x_test, y_test) #layer4
    error_est4       = kr.fit(x_train_part5, error_gt5).forward(x_test, y_test) #layer4
    
    result5           = expectation_inf + error_est1 + error_est2 + error_est3 + error_est4
    result4           = expectation_inf + error_est1 + error_est2 + error_est3
    result3           = expectation_inf + error_est1 + error_est2
    result2           = expectation_inf + error_est1
    result1           = expectation_inf
    
    
    eval_metrics = {}
    y_class = np.argmax(y_test, axis=-1)
    p_class1 = np.argmax(result1, axis=-1)
    p_class2 = np.argmax(result2, axis=-1)
    p_class3 = np.argmax(result3, axis=-1)
    p_class4 = np.argmax(result4, axis=-1)
    p_class5 = np.argmax(result5, axis=-1)
    
    eval_metrics['multiclass-acc of 1st layer'] = np.mean(y_class == p_class1)
    eval_metrics['multiclass-acc of 2nd layer'] = np.mean(y_class == p_class2)
    eval_metrics['multiclass-acc of 3rd layer'] = np.mean(y_class == p_class3)
    eval_metrics['multiclass-acc of 4th layer'] = np.mean(y_class == p_class4)
    eval_metrics['multiclass-acc of 5th layer'] = np.mean(y_class == p_class5)
    print(eval_metrics)
    
    
    
    print("KR including bandwith fitted in %.3f s" \
        % (time.time() - t0))
    
    ##### minist
    # n_class = 10
    # (x_train, y_train), (x_test, y_test) = mnist.load()
    # x_train, y_train, x_test, y_test = x_train.astype('float32'), \
    # y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')
    
    # # Fit regression models
    # t0 = time.time()
    # kr = KernelRegression(kernel="laplacian", gamma=0.5)
    # y_kr = kr.fit(x_train, y_train).forward(x_test, y_test) # X.shape: (100, 1), y.shape: (100,) np.expand_dims(y, axis=1).shape
    # print("KR including bandwith fitted in %.3f s" \
    #     % (time.time() - t0))