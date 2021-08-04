# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch import flatten
from torch import nn
import numpy as np
import torch
import os
import pickle

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2

from PIL import Image


########################

imgDirPath = 'dataset/images/'
saveModelPath = 'savedModels/'

data = []
targets = []
filenames = []
origSize = []

annotationPath = 'dataset/airplanes.csv'
rows = open(annotationPath).read().strip().split("\n")
for i,row in enumerate(rows):
    splitRow = row.split(',')
    filePath = imgDirPath + splitRow[0]
    h,w,c = cv2.imread(filePath).shape
    targetVal = splitRow[1:]
    origSize.append([w,h])

    targetVal[0] = int(targetVal[0])/w
    targetVal[1] = int(targetVal[1])/h
    targetVal[2] = int(targetVal[2])/w
    targetVal[3] = int(targetVal[3])/h

    filenames.append(filePath)
    targets.append(targetVal)



class BoundingBoxDataset(Dataset):
    def __init__(self, fileNameList, targetVec, imgDirPath,origSize,transform=None):
        self.fileNameList = fileNameList
        self.bboxCoord = torch.tensor(targetVec)
        self.transform = transform
        self.imgDir = imgDirPath
        self.origSize = torch.tensor(origSize)
    
    def getFileList(self):
        return self.fileNameList

    def __len__(self):
        return len(self.fileNameList)

    def __getitem__(self, idx):
        img_path = self.fileNameList[idx]
        imgSize = 256
        assert os.path.exists(img_path)
        image = Image.open(img_path).resize((imgSize,imgSize))
        label = self.bboxCoord[idx]
        label = torch.unsqueeze(label,0)

        origSizeVal = self.origSize[idx]

        if self.transform:
            image = self.transform(image)
            image = torch.unsqueeze(image,0)
        return image, label,origSizeVal



transformFun = transforms.Compose([
    transforms.ToTensor()
    ])





bbDataLoader = BoundingBoxDataset(filenames, targets,imgDirPath,origSize,transform=transformFun )
print(len(bbDataLoader))
img,bbox,origSize = bbDataLoader[10]
print(bbox)

splitLen = [int(len(bbDataLoader)*0.7),int(len(bbDataLoader)*0.3)]
train_loader, valid_loader = torch.utils.data.random_split(bbDataLoader, splitLen, generator=torch.Generator().manual_seed(42))


torch.save(valid_loader,"valid/valid_loader.pth")


# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')



from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # 8
        self.batchNorm = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3, padding=3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.conv2a = nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding=3)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=3)
        self.pool3 = nn.MaxPool2d(2,2)
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.conv3a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3)
        self.pool4 = nn.MaxPool2d(2,2)
        self.batchNorm4 = nn.BatchNorm2d(128)
        self.conv4a = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=3)
        self.pool5 = nn.MaxPool2d(2,2)
        self.batchNorm5 = nn.BatchNorm2d(256)
        self.conv5a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3)

        self.dropout = nn.Dropout(0.25)


        # Code to predict regression values

        self.fc1 = nn.Linear(50176, 100)
        self.fc2 = nn.Linear(100,4)


    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.batchNorm(self.conv1(x))))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = self.pool2(F.relu(self.conv2a(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = self.pool3(F.relu(self.conv3a(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        x = self.pool4(F.relu(self.conv4a(x)))
        x = F.relu(self.batchNorm5(self.conv5(x)))
        x = self.pool5(F.relu(self.conv5a(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = F.sigmoid(x)
        return x

# create a complete CNN
model = Net()
#print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

import torch.optim as optim

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



# number of epochs to train the model
#n_epochs = 8 # you may increase this number to train a final model
n_epochs = 100

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target,origSize in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        
        w,h = origSize

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        loss = criterion(output.float(), target.float())
        
        # backward pass: compute gradient of the loss with respect to model parameters
        # perform a single optimization step (parameter update)
        loss.backward()
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
    ######################    
    # validate the model #
    ######################
    model.eval()

    for data, target,origSize in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        
        w,h = origSize
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        #loss = criterion(output, target)
        loss = criterion(output.float(), target.float())

        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader)
    valid_loss = valid_loss/len(valid_loader)

    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    torch.save(model.state_dict(), saveModelPath+'model.pth')
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased          ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), saveModelPath+'model_valid.pth')
        valid_loss_min = valid_loss
