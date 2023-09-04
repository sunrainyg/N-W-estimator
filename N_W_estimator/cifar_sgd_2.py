import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import pdb
import torchvision
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pre_process(torchset,n_samples,num_classes=10):
    indices = list(np.random.choice(len(torchset),n_samples))

    trainset = []
    for ix in indices:
        x,y = torchset[ix]
        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        trainset.append(((x/np.linalg.norm(x)).reshape(-1),ohe_y))
    return trainset

def get_data(data_loader):
        X, y = [], []
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch
            X.append(inputs)
            y.append(labels)
        return torch.cat(X, dim=0), torch.cat(y, dim=0)
    
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)

    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class MSEloss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x, y):

        return torch.mean(torch.pow((x - y), 2))


class udfNetWork(nn.Module):
    
    def __init__(self, batch_size):
        super(udfNetWork, self).__init__()
        
        self.W              = nn.Parameter(torch.eye(3072, 3072, dtype=torch.float32), requires_grad=True)
        self.t              = nn.Parameter(torch.ones((batch_size,3072), dtype=torch.float32), requires_grad=True)
        self.c              = nn.Parameter(0.001 * torch.rand((1,10), dtype=torch.float32), requires_grad=True)
        self.bandwidth      = 1.
        self.iteration      = 1
        self.kernel         = lambda x, t: self.G_function(x, t, self.W, self.bandwidth, self.iteration)

    
    def euclidean_distances_M(self, samples, centers, M, squared=True):
        '''
        Calculate the Euclidean Distances between the samples and centers, using Ma distance
        squared = True: rbf
        squared = False: lap
        '''
        
        ## Obtain a vector containing the square norm value for each sample point.
        samples_norm2 = ((samples @ M) * samples).sum(-1)

        if samples is centers:
            centers_norm2 = samples_norm2
        else:
            centers_norm2 = ((centers @ M) * centers).sum(-1)
            
        distances = -2 * (samples @ M) @ centers.T
        distances = distances.add(samples_norm2.view(-1, 1))
        distances = distances.add(centers_norm2)
        if not squared:
            distances = distances.clamp(min=0).sqrt()
        return distances
    
    def G_function(self, samples, centers, M, gamma, iteration):

        assert gamma > 0
        kernel_mat = self.euclidean_distances_M(samples, centers, M, squared=True)
        # kernel_mat = kernel_mat.clamp(min=0) # Guaranteed non-negative
        gamma = 1. / gamma
        kernel_mat = kernel_mat.mul(-gamma) # point-wise multiply
        kernel_mat = kernel_mat.exp() #point-wise exp
        self.iteration += 1
        return kernel_mat
    
    def forward(self, samples):
        # print("self.W", self.W)
        kernel = self.kernel(samples.detach(), self.t)
        cc     = self.c.expand(samples.shape[0],10)
        result = kernel @ cc
        return result
    
    


if __name__ == "__main__":
    
    n_class         = 10
    batch_size      = 50000
    epoch           = 10
    
    transform       = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    trainset        = torchvision.datasets.CIFAR10(root='./data', train=True,
                    download=True, transform=transform)
    testset         = torchvision.datasets.CIFAR10(root='./data', train=False,
                    download=True, transform=transform)
    
    trainset        = pre_process(trainset, n_samples=50000, num_classes=10)
    trainloader     = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
    testset         = pre_process(testset, n_samples=10000, num_classes=10)
    testloader      = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

    model           = udfNetWork(batch_size).to(device)
    print("list(fc.named_parameters()):", list(model.named_parameters()))
    optimizer       = optim.Adam(model.parameters(), lr=0.1)
    # model.train()
    
    for _ in range(epoch):
        for batch_idx, (train_batch_x, train_batch_y) in enumerate(trainloader):
            
            x_train          = train_batch_x.view(batch_size, -1)
            # y_train          = to_categorical(train_batch_y, n_class) # label -> one hot
            y_train          = train_batch_y
            x_train, y_train = torch.tensor(x_train).to(device), torch.tensor(y_train).to(device)
            optimizer.zero_grad()
            output = model(x_train)
            criterion = MSEloss()
            # import pdb; pdb.set_trace()
            loss = criterion(output, y_train)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            print("one iteration finished, loss:{}".format(loss))


