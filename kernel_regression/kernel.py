'''Implementation of kernel functions.'''
import torch
import cifar
import pdb
import numpy as np

eps = 1e-12

def euclidean_distances(samples, centers, squared=True):
    '''Calculate the pointwise distance.

    Args:
        samples: of shape (n_sample, n_feature). := x
        centers: of shape (n_center, n_feature). := x_i
        squared: boolean.

    Returns:
        pointwise distances (n_sample, n_center).
    '''
    pdb.set_trace()
    samples_norm = np.sum(samples**2, axis=1) # center: torch.Size([49000, 3072]); samples: torch.Size([1758, 3072])
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = np.sum(centers**2, axis=1)
    centers_norm = np.reshape(centers_norm, (1, -1))

    distances = np.matmul(samples, centers.T) # distances: torch.Size([1758, 49000])
    distances *= -2
    distances += samples_norm
    distances += centers_norm
    
    if not squared:
        distances = np.maximum(distances, 0)
        distances = np.sqrt(distances)

    return distances


def gaussian(samples, centers, bandwidth):
    '''Gaussian kernel.
    
    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)
    kernel_mat = np.maximum(kernel_mat, 0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat += -gamma
    np.exp(kernel_mat, out=kernel_mat)
    return kernel_mat


def laplacian(samples, centers, bandwidth):
    '''Laplacian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


def dispersal(samples, centers, bandwidth, gamma):
    '''Dispersal kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.
        gamma: dispersal factor.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)
    kernel_mat.pow_(gamma / 2.)
    kernel_mat.mul_(-1. / bandwidth)
    kernel_mat.exp_()
    return kernel_mat


def ntk_relu(X, Z, depth=1, bias=0.):
    """
    Returns the evaluation of nngp and ntk kernels
    for fully connected neural networks
    with ReLU nonlinearity.
    
    depth  (int): number of layers of the network
    bias (float): (default=0.)
    """
    from torch import acos, pi
    kappa_0 = lambda u: (1-acos(u)/pi)
    kappa_1 = lambda u: u*kappa_0(u) + (1-u.pow(2)).sqrt()/pi
    Z = Z if Z is not None else X
    norm_x = X.norm(dim=-1)[:, None].clip(min=eps)
    norm_z = Z.norm(dim=-1)[None, :].clip(min=eps)
    S = X @ Z.T
    N = S + bias**2
    for k in range(1, depth):
        in_ = (S/norm_x/norm_z).clip(min=-1+eps,max=1-eps)
        S = norm_x*norm_z*kappa_1(in_)
        N = N * kappa_0(in_) + S + bias**2
    return N

def ntk_relu_unit_sphere(X, Z, depth=1, bias=0.):
    """
    Returns the evaluation of nngp and ntk kernels
    for fully connected neural networks
    with ReLU nonlinearity.
    Assumes inputs are normalized to unit norm.
    
    depth  (int): number of layers of the network
    bias (float): (default=0.)
    """
    from torch import acos, pi
    kappa_0 = lambda u: (1-acos(u)/pi)
    kappa_1 = lambda u: u*kappa_0(u) + (1-u.pow(2)).sqrt()/pi
    Z = Z if Z is not None else X
    S = X @ Z.T
    N = S + bias**2
    for k in range(1, depth):
        in_ = (S).clip(min=-1+eps,max=1-eps)
        S = kappa_1(in_)
        N = N * kappa_0(in_) + S + bias**2
    return N

if __name__ == "__main__":
    import torch
    from torch.nn.functional import normalize
    n, m, d = 1000, 800, 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.randn(n, d, device=DEVICE)
    X_ = normalize(X, dim=-1)
    Z = torch.randn(m, d, device=DEVICE)
    Z_ = normalize(Z, dim=-1)
    KXZ_ntk = ntk_relu(X, Z, 64, bias=1.)
    KXZ_ntk_ = ntk_relu_normalized(X_, Z_, 64, bias=1.)
    print(
        KXZ_ntk.diag().max().item(), 
        KXZ_ntk_.diag().max().item()
    )
