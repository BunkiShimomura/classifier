from create_dataset import MyDataset, MyNormalize, divide_dataset
from model import Net, train, test, learn, evaluate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from skimage import io
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

LABEL_IDX = 0
IMG_IDX = 1

csv_file_path = "/Users/Bunki/Desktop/PMP/code/pytorch_tutorial/script/classifier/data.csv"
ROOT_DIR = "/Users/Bunki/Desktop/PMP/code/pytorch_tutorial"

#print(pd.read_csv(csv_file_path))

imgDataset = MyDataset(csv_file_path, ROOT_DIR, transform=transforms.Compose([
    transforms.Resize(50),
    transforms.ToTensor(),
    MyNormalize()
    ]))


# Code to verify each shape or type
# Activate when something is wrong with original data
'''
print("________________")
print(len(imgDataset))
print(imgDataset[0])
print(type(imgDataset[0][0]))
print(imgDataset[0][0])
print(imgDataset[0][0][0].size)
print(imgDataset[0][0][0].shape)

trans = transforms.ToPILImage()
plt.imshow(trans(imgDataset[0][0]))
plt.gray()
plt.show()

print(type(trans(imgDataset[0][0])))
plt.imshow(trans(imgDataset[0][0]).convert('L'), cmap=plt.cm.binary)
plt.show()
'''

train_loader, test_loader, validation_loader = divide_dataset(imgDataset, 0.2, 16, 16)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

epoch = [1, 3, 5, 50, 100]
train_results = []
test_results = []
for ep in epoch:
    result = learn(train_loader, test_loader, ep)
    train_results.append(result[0])
    test_results.append(result[1])

print(train_results)

for result in train_results:
    print('epoch: ' + str(result[1]) + ", " + "Average loss: " + str(result[0]))

for result in test_results:
    print('epoch: ' + str(result[2]) + ", " + "Average loss: " + str(result[0]) + ", " + "Accuracy: " + str(result[1]))

print("evaluate")
evaluate(validation_loader, 'cnn_dict.model')
