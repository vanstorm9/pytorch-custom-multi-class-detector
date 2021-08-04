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

#tarImgPath = '../../dataset/images/airplane/image_0001.jpg'
#tarImgPath = '../../dataset/images/airplane/image_0799.jpg'
tarImgPath = '../../dataset/images/airplane/image_0081.jpg'

imgDirPath = '../../dataset/images/'

train_on_gpu = True

assert os.path.exists(modelPath)

classList = ['airplane','face','motorcycle']
numOfClass = len(classList)



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


imgOrig = cv2.imread(tarImgPath)
h,w,c = imgOrig.shape
img = cv2.resize(imgOrig,(256,256))
x = torch.tensor(img)/255.0
#x = x.permute(2,1,0)
x = x.permute(2,0,1)
x = torch.unsqueeze(x,0)
x = x.float().cuda()

outputBB,outputClass = model(x)


# Convert our torch tensor to numpy
predClass = torch.argmax(outputClass[0]).detach().cpu().numpy()
predName = classList[predClass]

print(predName,'    ', outputBB)
bb = outputBB.cpu().detach().numpy()[0]
pt0 = bb[:2]
pt1 = bb[2:]

pt0 = (int(pt0[0]*w),int(pt0[1]*h))
pt1 = (int(pt1[0]*w),int(pt1[1]*h))

fontScale = 0.5
imgOrig = cv2.putText(imgOrig, predName, (pt0[0],pt0[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale, (0,255,0), 2, cv2.LINE_AA)

cv2.rectangle(imgOrig,pt0,pt1,(0,255,0),3)

cv2.imshow('result',imgOrig)
cv2.waitKey(0)

