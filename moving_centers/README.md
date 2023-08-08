# Radial-Basis-Function-RBF

This implementation makes use of Radial Basis function with 100 centers for MNIST digit dataset classification that has the images of digits from 0 to 9. 


__Radial basis function (RBF)__ is a type of machine learning algorithm that is commonly used for classification and function approximation. RBF is a function whose value depends only on the distance between the input data and a set of fixed points, called centers.
# Testing Accuracy : 91.58 %
# 1. Dataset
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. It is a dataset of 70,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9. It has a training set of 60,000 examples, and a test set of 10,000 examples.

# 2. Results
## 2.1 Training vs Testing loss
![5e4f2927-df18-4c05-ad50-8e12f815a92e](https://user-images.githubusercontent.com/34654665/224183831-bed171e5-2dad-4004-803f-6eb0315c4885.png)

## 2.2 Training vs Testing accuracy
![3e1d365d-d22c-45c9-9010-edca4f4851c8](https://user-images.githubusercontent.com/34654665/224183871-c9398ba3-db52-4986-a18c-f6434ed5d879.png)

## 2.3 Confusion Matrix
![314965ca-4ee3-4268-8f7f-6df22e9ee82e](https://user-images.githubusercontent.com/34654665/224183898-1993ac71-0bc7-48d2-8f44-d6edc16a5656.png)


## 2.4 Classification Report
![image](https://user-images.githubusercontent.com/34654665/224183950-43da69a1-ba1a-44db-aeea-f2a3ce866a4a.png)

## 2.5. Varying the number of cluster centers
If we vary the number of cluster centers like [10, 30, 50, 100], we can get the imporved performances. With the increase in number of RBFs centers, the accuracy of the model increases which can be seen in the below figure. 

![07ee11ed-8fdf-4082-b0ad-feb8f8b1a841](https://user-images.githubusercontent.com/34654665/224184415-59819579-be73-42eb-a8fa-8037c2b67d96.png)

## 2.6. Use multiple sigma instead of single sigma
Sigma parameter in RBF network controls the width of RBF, which affects how sensitive the network is to changes in the input data. Having multiple sigma values in an RBF net can help improve the performance allowing it to capture different details from the input data. I have used a set of predefined sigma values of 0.5, 1, 2.0 and 5.0 to see the effect of the performance of the model and obtain an accuracy of 90.85% which increases by 0.61% from having a sigma value of 5. But using multiple sigma values can increase the complexity of the RBF network.

![7c90eed7-8092-4fb6-998e-609950db836a](https://user-images.githubusercontent.com/34654665/224184547-ebd10b61-3d10-4211-862a-4e6493b30cb4.png)
![b84ef06c-0f9c-44d5-96ae-1e18d4ddd6b4](https://user-images.githubusercontent.com/34654665/224184556-4a2a3d63-32fb-4065-b47e-c9da43ac0204.png)
