import os
import torchvision
import torchvision.transforms as transforms

## CIFAR10
download_dir = "./data"
if not os.path.exists(download_dir):
    os.makedirs(download_dir)


if not os.path.exists(download_dir):
    os.makedirs(download_dir)


transform = transforms.Compose(
    [transforms.ToTensor(),  # 将 PIL 图像或 numpy 数组转换为 PyTorch Tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 对图像进行归一化


trainset = torchvision.datasets.CIFAR10(root=download_dir, train=True,
                                        download=True, transform=transform)


testset = torchvision.datasets.CIFAR10(root=download_dir, train=False,
                                       download=True, transform=transform)

print("CIFAR-10 dataset downloaded and saved to the new directory.")
