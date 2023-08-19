import torch.nn.functional as F
import torch
import torchvision
import cifar, mnist, svhn
import math

class ConvHyperBF():
    def __init__(self, trainloader):
        super(ConvHyperBF, self).__init__()
        self.x_train        = trainloader
        self.M              = torch.eye(self.x_train.shape[-1])
        self.M              = self.M.to('cuda')
        self.kernel_cpu     = lambda x, z: self.laplacian_M(x, z, self.M.cpu(), self.gamma)
        self.trainloader    = trainloader
    
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

        centers_term = ()
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
        kernel_mat              = self.euclidean_distances_M(samples, centers, M, squared=False)
        kernel_mat.clamp_(min=0) # Guaranteed non-negative
        gamma                   = 1. / gamma
        kernel_mat.mul_(-gamma) # point-wise multiply
        kernel_mat.exp_() #point-wise exp
        return kernel_mat
            
    def euclidean_distances_M(self, samples, centers, M, squared=False):
        '''
        Calculate the Euclidean Distances between the samples and centers, using Ma distance
        squared = True: rbf
        squared = False: lap
        '''
        
        ## Obtain a vector containing the square norm value for each sample point.
        samples_norm2           = ((samples @ M) * samples).sum(-1) # torch.Size([1])

        if samples is centers:
            centers_norm2       = samples_norm2
        else:
            centers_norm2       = ((centers @ M) * centers).sum(-1) # torch.Size([1])
            
        distances               = -2 * (samples @ M) @ centers.T # torch.Size([10000, 10000])
        distances.add_(samples_norm2.view(-1, 1))
        distances.add_(centers_norm2)

        if not squared:
            distances.clamp_(min=0).sqrt_()
        return distances

    def conv2d_like_hyperBF(self, input, kernel, bias=0, stride=1, padding=0):
        '''
        input: x-x_i --- x and x_i indicates different part of the datasets
        kernel: M ---  M is learned from data
        
        '''
        
        if padding > 0:
            input = F.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0))
            
        bs, in_channel, input_h, input_w                = input.shape
        out_channel, in_channel, kernel_h, kernel_w     = kernel.shape
        
        if bias is None:
            bias = torch.zeros(out_channel)
            
        output_h                                        = (math.floor((input_h - kernel_h)/stride)+1)
        output_w                                        = (math.floor((input_w - kernel_w)/stride)+1)
        output                                          = torch.zeros(bs, out_channel, output_h, output_w)
        
        for ind in range(bs):
            for oc in range(out_channel):
                for ic in range(in_channel):
                    for i in range(0, input_h-kernel_h+1, stride):
                        for j in range(0, input_w-kernel_w+1, stride):
                            region                      = input[ind, ic, i:i+kernel_h, j:j+kernel_w]
                            # output[ind, oc, int(i/stride), int(j/stride)] += torch.sum(region * kernel[oc, ic])
                            matrix                      = self.kernel_cpu(self.train_x, self.test_x)
                            
                            
                            output[ind, oc, int(i/stride), int(j/stride)] += euclidean_distances_M(input[ind],input[ind])
                output[ind, oc] += bias[oc]
        
        return output

    def fit(self, x):
        
        if self.train_M_x is not None:
            epochs = 5
            for epoch in range(epochs):
                alpha           = self.fit_predictor_lstsq(self.train_M_x, self.train_M_y) #alpha.shape: torch.Size([30000, 10])
                self.M          = self.update_M(self.train_M_x, alpha)
                
                print("One round finished")
                
        for i, data in enumerate(self.trainloader, 0):
            inputs, labels = data
        x = self.conv2d_like_hyperBF(input, kernel)
        ## concat x
        
        return x
        

if __name__ == "__main__":
    
    # input = torch.randn(2,3,5,5) # bs * in_channel * in_h * in_w
    # kernel = torch.randn(3,3,3,3) # out_channel * in_channel * kernel_h * kernel_w
    # bias = torch.randn(3)
    
    # pytorch_conv2d_api_output = F.conv2d(input, kernel, bias=bias, padding=1, stride=2)
    # mm_conv2d_full_output = matrix_multiplication_for_conv2d(input, kernel, bias=bias, padding=1, stride=2)
    # flag = torch.allclose(pytorch_conv2d_api_output, mm_conv2d_full_output)
    # print("all_close", flag)
    trainset = torchvision.datasets.CIFAR10(root='/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/data/', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    # fit
    net = ConvHyperBF(trainloader,)
    
    # inference
    correct = 0
    total = 0
    with torch.no_grad():  # 在推理过程中不需要计算梯度
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 找到最大预测值的索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
    

    