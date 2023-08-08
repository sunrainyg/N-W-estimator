"""
Author: Md Mahedi Hasan 
For Homework Assingment 12 (CpE 520)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision
import torch 
from torch import nn 
from sklearn.cluster import KMeans
import pickle 
from sklearn.metrics import confusion_matrix
import seaborn as sns 

data_dir = "./data"
saved_image_dir = "./images"      
num_epochs = 300
batch_size = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
num_centers = 10
lr = 0.005
multi_gpu = True
is_load = False   
is_save = True 
model_file = "rbf_pytorch.pth"
init_kmeans = True 

transform       = transforms.Compose(
                    [transforms.ToTensor()])

train_data = datasets.MNIST(root='./data', train=True, transform=transform)

numpy_array = train_data.data.reshape(-1, 784)  / 255.0
tensor_float = torch.FloatTensor(numpy_array)
tensor_double = tensor_float.double()
X_train = tensor_double.to(device)

# Cluster the data using KMeans with k=100
kmeans = KMeans(n_clusters=30)
kmeans.fit(X_train.cpu().numpy())

# find the cluster centers
clusters = kmeans.cluster_centers_.astype(float)
print(clusters.shape)

class RBFNet(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.num_class = num_class
        self.num_centers = num_centers


        if init_kmeans == True: 
            # centroids saved from rbf psuedo inverse method
            self.centers = torch.from_numpy(clusters)

            self.centers = nn.Parameter(self.centers).type(torch.float32).to(device)

        else: 
            # standard gaussian
            self.centers = nn.Parameter(torch.rand(self.num_centers, input_dim))


        self.beta = nn.Parameter(torch.ones(1,self.num_centers)/10)
        self.linear = nn.Linear(self.num_centers, self.num_class, bias=True)

    def rbf_layer(self, x):
        n_input = x.size(0) 
        A = self.centers.view(self.num_centers,-1).repeat(n_input,1,1)
        B = x.view(n_input,-1).unsqueeze(1).repeat(1,self.num_centers,1)
        C = torch.exp(-self.beta.mul((A-B).pow(2).sum(2,keepdim=False).sqrt())) # exp(-B ||c - x||)
        return C

    def forward(self, x):
        radial_val = self.rbf_layer(x)
        class_score = self.linear(radial_val)
        return class_score


class Train:
    def __init__(self):
        mean = (0.1307, )
        std = (0.3081, ) 
        
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_dataset = torchvision.datasets.MNIST(
                            root=data_dir, 
                            train=True, 
                            download=True,
                            transform=trans)
        self.train_loader = DataLoader(train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True, num_workers=6)

        val_dataset = torchvision.datasets.MNIST(
                            root=data_dir, 
                            train=False, 
                            download=True,
                            transform=trans)
        self.val_loader = DataLoader(val_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True, num_workers=6)
        self.model = RBFNet().to(device)

        if multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-5)
        self.schedular = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.90)

        self.ls_train_loss = []
        self.ls_train_acc = []
        self.ls_val_loss = []
        self.ls_val_acc = []

        if is_load:
            print("loading ...")
            checkpoint = torch.load(model_file)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"]) 



    def validate(self):
        total_loss = 0
        result = 0
        for img, label in self.val_loader:
            img = img.view(-1, 784).to(device)
            label = label.type(torch.long).to(device)

            scores = self.model(img)
            loss = self.loss(scores, label)

            # calculate loss & result 
            output = torch.argmax(scores, dim=1)
            total_loss += loss.item()
            output = output.cpu().detach().numpy() 
            label = label.cpu().detach().numpy()
            result += sum(output[i] == label[i] for i in range(len(label)))

        acc = result / (len(self.val_loader) * batch_size)

        epoch_loss = total_loss / (len(self.val_loader) * batch_size)
        self.ls_val_loss.append(epoch_loss)
        self.ls_val_acc.append(acc)
        print("val loss {:.7f} | val accuracy: {:.7f}".format(epoch_loss, acc))


    def train(self):
        for epoch in range(num_epochs):
            total_loss = 0
            result = 0
            for img, label in self.train_loader:
                img = img.view(-1, 784).to(device)
                label = label.type(torch.long).to(device)
        
                self.optimizer.zero_grad()
                scores = self.model(img)
                loss = self.loss(scores, label)
                loss.backward()
                self.optimizer.step()
                
                # calculate loss & result 
                output = torch.argmax(scores, dim=1)
                total_loss += loss.item()
                output = output.cpu().detach().numpy() 
                label = label.cpu().detach().numpy()
                result += sum(output[i] == label[i] for i in range(len(label)))

            if (epoch % 60 == 0 and epoch !=0):
                self.schedular.step() 
                print(self.schedular.get_lr())

            # change according to your experiment
            if (epoch % 5 == 0 and epoch !=0):
                acc = result / (len(self.train_loader) * batch_size)

                epoch_loss = total_loss / (len(self.train_loader) * batch_size)
                self.ls_train_loss.append(epoch_loss)
                self.ls_train_acc.append(acc)
                print("Epoch {} | train loss {:.7f} | train accuracy: {:.7f}".format(
                                                            epoch, epoch_loss, acc))

                self.validate()


        if is_save:
            checkpoint = {}
            checkpoint["model"] = self.model.state_dict() 
            checkpoint["optimizer"] = self.optimizer.state_dict()
            torch.save(checkpoint, model_file)
            print("saving model")

            with open("train_acc", "wb") as ta:
                pickle.dump(self.ls_train_acc, ta)

            with open("train_loss", "wb") as tl:
                pickle.dump(self.ls_train_loss, tl)

            with open("val_acc", "wb") as va:
                pickle.dump(self.ls_val_acc, va)

            with open("val_loss", "wb") as vl:
                pickle.dump(self.ls_val_loss, vl)


    def plot_learning_curve(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        epochs = [i for i in range(5, num_epochs, 5)]
        d_set, e_type = filename.split("_")
        t1 = ("Validation" if d_set == "val" else "Train")
        t2 = (" accuracy" if e_type == "acc" else " loss")

        plt.plot(epochs, data, 'g', label= t1 + t2)
        plt.title(t1 + t2 + " on MNIST dataset")
        plt.xlabel('Number of Epochs')
        plt.ylabel(t2.capitalize())
        plt.legend()
        plt.show()

    def draw_confusion_matrix(self):
        self.model.eval()
        true_labels = []
        pred_labels = []

        for img, label in self.val_loader:
            img = img.to(device) 
            scores = self.model(img)
            output = torch.argmax(scores, dim=1).cpu().detach().tolist()
            pred_labels += output 
            true_labels += label

        # make confusion matrix 
        c_matrix = confusion_matrix(y_true=true_labels, y_pred=pred_labels)

        plt.figure(figsize = (10, 10))
        sns. set(font_scale=1.4)
        sns.heatmap(c_matrix, annot=True, fmt = 'g', linewidths=.5)

        # labels, title 
        plt.xlabel('Predicted Label', fontsize=10, labelpad=11)
        plt.ylabel('True Label', fontsize=10)
        plt.show()

if __name__ == "__main__":
    t = Train()
    # filename = "train_acc"
    # t.plot_learning_curve(filename)
    #t.draw_confusion_matrix()
    t.train()