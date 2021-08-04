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

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import torchvision.models as models

from PIL import Image


########################

#modelPath = 'model_valid.pth'
modelPath = 'savedModels/model_valid.pth'
#modelPath = 'savedModels/model.pth'


imgDirPath = '../../dataset/images/'

dataloaderPath = 'valid/valid_loader.pth'

classList = ['airplane','face','motorcycle']
numOfClass = len(classList)


train_on_gpu = True

assert os.path.exists(modelPath)

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
        label = torch.unsqueeze(label,0)

        origSizeVal = self.origSize[idx]
        classVal = self.classVec[idx]
        classVal = torch.unsqueeze(classVal,0)

        if self.transform:
            image = self.transform(image)
            image = torch.unsqueeze(image,0)
        return image, label,classVal,origSizeVal


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
        #print('Got passed this line')
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

# load model weights
model.load_state_dict(torch.load(modelPath))
model.eval()


if train_on_gpu:
    model.cuda()

valid_loader = torch.load(dataloaderPath)

# Iterate through our validation dataset to make predictions
for data, targetBB,targetClass,origSize in valid_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, targetBB, targetClass = data.cuda(), targetBB.cuda(), targetClass.cuda()
    #output = model(data)
    outputBB,outputClass = model(data)

    #print(outputClass)
    predClass = torch.argmax(outputClass[0]).detach().cpu().numpy()
    predName = classList[predClass]
    

    
    img = data[0].detach().permute(1,2,0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    origSize = origSize.detach().cpu().numpy()
    w,h = origSize

    imgNormSize = cv2.resize(img,(w,h))


    # We then draw bounding boxes
    bb = outputBB.cpu().detach().numpy()[0]

    print(predName, '      ', bb)
    
    pt0 = bb[:2]
    pt1 = bb[2:]

    pt0 = (int(pt0[0]*w),int(pt0[1]*h))
    pt1 = (int(pt1[0]*w),int(pt1[1]*h))

    fontScale = 0.5
    
    cv2.rectangle(imgNormSize,pt0,pt1,(0,255,0),3)
    imgNormSize = cv2.putText(imgNormSize, predName, (pt0[0],pt0[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow('res',imgNormSize)
    cv2.waitKey(0)
