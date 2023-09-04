import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 设置随机种子以便复现结果
torch.manual_seed(42)

class GaussianActivation(nn.Module):
    def __init__(self, mu=0, sigma=1):
        super(GaussianActivation, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        # 高斯激活函数的实现
        return torch.exp(-2 * x)
    
# 定义简单的卷积神经网络模型
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.gaussian_activation = GaussianActivation(mu=0, sigma=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)  # 输入尺寸经过卷积和池化后变为 16x16x16

    def forward(self, x):
        
        x = -2 * (x * x)
        x = self.pool(self.gaussian_activation(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)  # 展平卷积层输出以便连接全连接层
        # x = self.fc1(x)
        return x

# preprocess

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 数据标准化
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
net = SimpleConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练网络
for epoch in range(5):  # 遍历数据集多次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # 梯度清零
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 优化
        
        running_loss += loss.item()
        if i % 200 == 199:  # 每200个小批次打印一次损失
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

# 加载CIFAR-10测试数据集
testset = torchvision.datasets.CIFAR10(root='/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 在测试集上进行推理
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

