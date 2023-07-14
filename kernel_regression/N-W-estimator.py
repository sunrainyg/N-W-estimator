## copyright Yulu Gan 2023.

import numpy as np
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
            self.gamma = gamma
            self.block = block

    import numpy as np

    def normalize_rows(self, matrix):
        row_sums = matrix.sum(axis=1)  # 求每一行的总和
        normalized_matrix = matrix / row_sums[:, np.newaxis]  # 将每一行元素除以该行总和
        return normalized_matrix

    def fit(self, x_train, y_train):
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

        K_test          = pairwise_kernels(self.x_train, x_test, metric=self.kernel, gamma=self.gamma) # K.shape (49000, 10000)
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
    cifar10_dir = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/dataset/cifar10/cifar-10-batches-py'
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
    kr = KernelRegression(kernel="rbf", gamma=0.1)
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
    
    
    ## inference
    expectation_inf  = kr.fit(x_train_part1, y_train_part1).forward(x_test, y_test) #layer1
    error_est1       = kr.fit(x_train_part2, error_gt2).forward(x_test, y_test) #layer2
    error_est2       = kr.fit(x_train_part3, error_gt3).forward(x_test, y_test) #layer3
    error_est3       = kr.fit(x_train_part4, error_gt4).forward(x_test, y_test) #layer4
    error_est4       = kr.fit(x_train_part5, error_gt4).forward(x_test, y_test) #layer4
    
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