try:
    from eigenpro3.models import KernelModel
    EIGENPRO_AVAILABLE = True
except ModuleNotFoundError:
    print('`eigenpro2` not installed.. using torch.linalg.solve for training kernel model')
    EIGENPRO_AVAILABLE = False
    
import torch, numpy as np
from .kernels import laplacian_M, euclidean_distances_M
from tqdm import tqdm
import hickle

class RecursiveFeatureMachine(torch.nn.Module):

    def __init__(self, device=torch.device('cpu'), mem_gb=32, diag=False, centering=False, reg=1e-3):
        super().__init__()
        self.M = None
        self.model = None
        self.diag = diag # if True, Mahalanobis matrix M will be diagonal
        self.centering = centering # if True, update_M will center the gradients before taking an outer product
        self.device = device
        self.mem_gb = mem_gb
        self.reg = reg # only used when fit using direct solve

    def get_data(self, data_loader):
        X, y = [], []
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch
            X.append(inputs)
            y.append(labels)
        return torch.cat(X, dim=0), torch.cat(y, dim=0)

    def update_M(self):
        raise NotImplementedError("Must implement this method in a subclass")

    def fit_predictor(self, centers, targets, **kwargs):
        '''
        input: center  - X_train 
               targets - y_train
        self.M - create a unit matrix
        self.weight - alpha
        '''
        
        self.centers = centers
        
        if self.M is None:
            ## First time into here to initialze the predictor
            if self.diag:
                self.M = torch.ones(centers.shape[-1], device=self.device)
            else:
                self.M = torch.eye(centers.shape[-1], device=self.device)
                
        if self.fit_using_eigenpro and EIGENPRO_AVAILABLE:
            self.weights = self.fit_predictor_eigenpro(centers, targets, **kwargs)
        else:
            ## here
            self.weights = self.fit_predictor_lstsq(centers, targets)

    def fit_predictor_lstsq(self, centers, targets):
        '''
        Function: solve the alpha
        Equation: alpha * K(X,X) = Y
        Return: alpha
        '''
        
        return torch.linalg.solve(
            self.kernel(centers, centers) 
            + self.reg*torch.eye(len(centers), device=centers.device), 
            targets
        )

    def fit_predictor_eigenpro(self, centers, targets, **kwargs):
        n_classes = 1 if targets.dim()==1 else targets.shape[-1]
        self.model = KernelModel(self.kernel, centers, n_classes, device=self.device)
        _ = self.model.fit(centers, targets, mem_gb=self.mem_gb, **kwargs)
        return self.model.weights

    def predict(self, samples):
        '''
        Function: predict
        Detail:
        [self.kernel : (laplacian_M, input: x, z)] @ self.weights
        '''
        return self.kernel(samples, self.centers) @ self.weights

    def fit(self, train_loader, test_loader,
            iters=3, name=None, reg=1e-3, method='lstsq', 
            train_acc=False, loader=True, classif=True):
        '''
        Function:
        
        '''
        
        if method=='eigenpro':
            raise NotImplementedError(
                "EigenPro method is not yet supported. "+
                "Please try again with `method='lstlq'`")
            #self.fit_using_eigenpro = (method.lower()=='eigenpro')
        self.fit_using_eigenpro = False
        
        if loader:
            print("Loaders provided")
            X_train, y_train = self.get_data(train_loader) #y_train: torch.Size([20000, 10])
            X_test, y_test = self.get_data(test_loader)
            print("y_test.shape", y_test.shape)
            print("Dataset shape: x_train - {}, y_train - {}, x_test - {}, y_test - {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
            
        else:
            X_train, y_train = train_loader
            X_test, y_test = test_loader
            print("Dataset shape: x_train - {}, y_train - {}, x_test - {}, y_test - {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
        
        
        for i in range(iters):
            ## initiate self.M and calculate self.weight
            self.fit_predictor(X_train, y_train)
            
            if classif:
                train_acc = self.score(X_train, y_train, metric='accuracy')
                print(f"Round {i}, Train Acc: {train_acc:.2f}%")
                test_acc = self.score(X_train, y_train, metric='accuracy')
                print(f"Round {i}, Test Acc: {test_acc:.2f}%")

            test_mse = self.score(X_test, y_test, metric='mse')
            print(f"Round {i}, Test MSE: {test_mse:.4f}")
            
            self.update_M(X_train)

            if name is not None:
                hickle.dump(self.M, f"saved_Ms/M_{name}_{i}.h")

        self.fit_predictor(X_train, y_train)
        final_mse = self.score(X_test, y_test, metric='mse')
        print(f"Final MSE: {final_mse:.4f}")
        if classif:
            final_test_acc = self.score(X_test, y_test, metric='accuracy')
            print(f"Final Test Acc: {final_test_acc:.2f}%")
            
        return self.M, final_mse
    
    def score(self, samples, targets, metric='mse'):
        '''
        Function: calculate the score
        '''
        ##
        preds = self.predict(samples)  #preds: torch.Size([20000, 10]); sampels: torch.Size([20000, 3072])
        if metric=='accuracy':
            return (1.*(targets.argmax(-1) == preds.argmax(-1))).mean()*100.# targets: torch.Size([20000, 10])
        elif metric=='mse':
            return (targets - preds).pow(2).mean()


class LaplaceRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel = lambda x, z: laplacian_M(x, z, self.M, self.bandwidth) # must take 3 arguments (x, z, M)
        
    def update_M(self, samples):
        '''
        Input:  samples - X_train; same as centers
                self.weights - alpha
                
        Notion: K := K_M(X,X) / ||x-z||_M
                samples_term := Mx * K_M(x,z)
                centers_term := Mz * K_M(x,z)
                
        Equation: grad(K_M(x,z)) = (Mx-Mz) * (K_M(x,z)) / (L||x-z||_M),
        where x is samples, z is center,
        
        '''
        
        K = self.kernel(samples, self.centers) # K_M(X, X)

        dist = euclidean_distances_M(samples, self.centers, self.M, squared=False) # Ma distance
        dist = torch.where(dist < 1e-10, torch.zeros(1, device=dist.device).float(), dist) # Set those small values to 0 to avoid errors caused by dividing by too small a number.

        K = K/dist # K_M(X,X) / M_distance
        K[K == float("Inf")] = 0. #avoid infinite big values

        p, d = self.centers.shape
        p, c = self.weights.shape
        n, d = samples.shape
        
        samples_term = (
                K # (n, p)
                @ self.weights # (p, c)
            ).reshape(n, c, 1) ## alpha * K_M(X,X) / M_distance
        
        if self.diag:
            centers_term = (
                K # (n, p)
                @ (
                    self.weights.view(p, c, 1) * (self.centers * self.M).view(p, 1, d)
                ).reshape(p, c*d) # (p, cd)
            ).view(n, c, d) # (n, c, d)

            samples_term = samples_term * (samples * self.M).reshape(n, 1, d)
            
        else:       
            ## Here ## 
            centers_term = (
                K # (n, p)
                @ (
                    self.weights.view(p, c, 1) * (self.centers @ self.M).view(p, 1, d)
                ).reshape(p, c*d) # (p, cd)
            ).view(n, c, d) # (n, c, d) ## alpha * K_M(X,X) / M_distance

            samples_term = samples_term * (samples @ self.M).reshape(n, 1, d) ## Mx * alpha * K_M(X,X) / M_distance

        G = (centers_term - samples_term) / self.bandwidth # (n, c, d)
        
        if self.centering:
            G = G - G.mean(0) # (n, c, d)
        
        if self.diag:
            torch.einsum('ncd, ncd -> d', G, G)/len(samples)
        else:
            self.M = torch.einsum('ncd, ncD -> dD', G, G)/len(samples)
            print("self.M.shape:", self.M.shape)



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    
    # define target function
    def fstar(X):
        return torch.cat([
            (X[:, 0]  > 0)[:,None],
            (X[:, 1]  < 0.1)[:,None]],
            axis=1)


    # create low rank data
    n = 4000
    d = 100
    np.random.seed(0)
    X_train = torch.from_numpy(np.random.normal(scale=0.5, size=(n,d)))
    X_test = torch.from_numpy(np.random.normal(scale=0.5, size=(n,d)))

    y_train = fstar(X_train).double()
    y_test = fstar(X_test).double()
    import pdb; pdb.set_trace()
    model = LaplaceRFM(bandwidth=1., diag=False, centering=False)
    model.fit(
        (X_train, y_train), 
        (X_test, y_test), 
        loader=False,
        iters=5,
        classif=False
    ) 
