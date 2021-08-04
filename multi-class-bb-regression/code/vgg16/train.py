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
import torchvision.models as models
import cv2
import math

from PIL import Image


########################

imgDirPath = '../../dataset/images/'
saveModelPath = 'savedModels/'

data = []
targets = []
filenames = []
classList = []
origSize = []

className = set()

annotationRoot = '../../dataset/annotations/'
annotationList = ['airplane.csv','face.csv','motorcycle.csv']

# Create list for our annotations to be fed into our datalaoder
for classInt,annotationFile in enumerate(annotationList):
    annotationPath = annotationRoot + annotationFile
    
    assert os.path.exists(annotationPath)
    classType = annotationFile.split(".csv")[0]
    className.add(classType)

    rows = open(annotationPath).read().strip().split("\n")
    for i,row in enumerate(rows):
        splitRow = row.split(',')
        filePath = imgDirPath + classType +'/'+splitRow[0]
        assert os.path.exists(filePath)
        h,w,c = cv2.imread(filePath).shape
        targetVal = splitRow[1:5]
        
        origSize.append([w,h])
        classList.append(classInt)

        targetVal[0] = int(targetVal[0])/w
        targetVal[1] = int(targetVal[1])/h
        targetVal[2] = int(targetVal[2])/w
        targetVal[3] = int(targetVal[3])/h


        filenames.append(filePath)
        targets.append(targetVal)


numOfClass = len(className)


class BoundingBoxDataset(Dataset):
    def __init__(self, fileNameList, targetVec, imgDirPath,origSize,classVec,transform=None):
        self.fileNameList = fileNameList
        self.bboxCoord = torch.tensor(targetVec)
        self.transform = transform
        self.imgDir = imgDirPath
        self.origSize = torch.tensor(origSize)
        self.classVec = torch.tensor(classVec)
    
    def getFileList(self):
        return self.fileNameList

    def __len__(self):
        return len(self.fileNameList)

    def __getitem__(self, idx):
        img_path = self.fileNameList[idx]
        imgSize = 256
        assert os.path.exists(img_path)
        image = Image.open(img_path).resize((imgSize,imgSize)).convert('RGB')
        label = self.bboxCoord[idx]

        origSizeVal = self.origSize[idx]
        classVal = self.classVec[idx]

        if self.transform:
            image = self.transform(image)
        return image, label,classVal,origSizeVal

# Allow us to convert our images to tensors
transformFun = transforms.Compose([
    transforms.ToTensor()
    ])



bbDataLoader = BoundingBoxDataset(filenames, targets,imgDirPath,origSize,classList,transform=transformFun )
print('Length of data: ', len(bbDataLoader))
img,bbox,origSize,classVal = bbDataLoader[10]

splitLen = [math.ceil(len(bbDataLoader)*0.7),math.floor(len(bbDataLoader)*0.3)]

train_loader, valid_loader = torch.utils.data.random_split(bbDataLoader, splitLen, generator=torch.Generator().manual_seed(32))

# Save our valid dataset to test in another script
torch.save(valid_loader,"valid/valid_loader.pth")

train_loader = DataLoader(train_loader, batch_size=64, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_loader, batch_size=64, shuffle=True, num_workers=4)


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

        self.dropout = nn.Dropout(0.25)
        self.vgg16 = models.vgg16(pretrained=True)
        
        # Here we write code to predict regression values

        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100,4)

        # Code to predict the class
        self.softmaxFc1 = nn.Linear(1000,100) 
        self.softmaxFc2 = nn.Linear(100,numOfClass)


    def forward(self, x):
        # add sequence of convolutional and max pooling layers

        x = self.vgg16(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        # Predict bounding box
        xBB = F.relu(self.fc1(x))
        xBB = self.dropout(xBB)
        xBB = F.relu(self.fc2(xBB))

        # Predict class
        xClass = F.relu(self.softmaxFc1(x))
        xClass = self.dropout(xClass)
        xClass = F.relu(self.softmaxFc2(xClass))

        return xBB,xClass
        
# create a complete CNN
model = Net()
#print(model)
# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()


import torch.optim as optim

criterionClass = nn.CrossEntropyLoss()
#criterionBB = nn.NLLLoss()
criterionBB = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



# number of epochs to train the model
n_epochs = 200

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, targetBB,targetClass,origSize in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, targetBB,targetClass = data.cuda(), targetBB.cuda(),targetClass.cuda()

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        outputBB,outputClass = model(data)
        
        lossClass = criterionClass(outputClass, targetClass)
        lossBB = criterionBB(outputBB.float(), targetBB.float())

        loss = lossBB + lossClass

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

    for data, targetBB,targetClass,origSize in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, targetBB,targetClass = data.cuda(), targetBB.cuda(),targetClass.cuda()
        
        #w,h = origSize
        
        # forward pass: compute predicted outputs by passing inputs to the model
        outputBB,outputClass = model(data)
        # calculate the batch loss
        lossClass = criterionClass(outputClass, targetClass)
        lossBB = criterionBB(outputBB.float(), targetBB.float())

        loss = lossClass + lossBB

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
