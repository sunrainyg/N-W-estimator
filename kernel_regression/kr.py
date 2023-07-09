import numpy as np
import pdb
import time
import kernel
import cifar
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

    def __init__(self, kernel_fn, block=1, kernel="rbf", gamma=None):
            self.kernel_fn = kernel_fn
            self.kernel = kernel
            self.gamma = gamma
            self.block = block

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
        K_train = pairwise_kernels(self.x_train, self.x_train, metric=self.kernel, gamma=self.gamma)
        # K_train = self.kernel_fn(self.x_train, self.x_train)
        alpha   = np.linalg.solve(K_train, self.y_train) # K_train: (49000, 49000); self,y_train: (49000, 10); alpha:(49000, 10)
        K_test  = pairwise_kernels(self.x_train, x_test, metric=self.kernel, gamma=self.gamma) # K.shape (49000, 10000)
        # K_test  = self.kernel_fn(self.x_train, x_test)
        output  = np.matmul(K_test.T, alpha) # K_test:(10000, 49000), alpha:(49000, 10)
        
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
    n_class = 10
    cifar10_dir = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/dataset/cifar10/cifar-10-batches-py'
    (x_train, y_train), (x_test, y_test) = cifar.load(cifar10_dir)
    x_train, y_train, x_test, y_test = x_train.astype('float32'), \
    y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')
    
    kernel_fn = lambda x,y: kernel.gaussian(x, y, bandwidth=5)
    
    # Fit regression models
    t0 = time.time()
    kr = KernelRegression(kernel_fn, kernel="rbf", gamma=0.02)
    # y_kr = kr.fit(X, y).forward(X) # X.shape: (100, 1), y.shape: (100,) np.expand_dims(y, axis=1).shape
    y_kr = kr.fit(x_train, y_train).forward(x_test, y_test) # X.shape: (100, 1), y.shape: (100,) np.expand_dims(y, axis=1).shape
    print("KR including bandwith fitted in %.3f s" \
        % (time.time() - t0))