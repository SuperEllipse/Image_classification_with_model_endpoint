## Vish: This is a workaround for using Jupyter runtime , since i am facing issues in deploying an inferencer for 
##       PBJ Runtime
import subprocess
import sys

install_packages=["torch","torchvision", "torchaudio", "torchinfo", "opencv-python-headless" , "tqdm", "matplotlib"]
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
for package in install_packages:
  install(package)
 

import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np 
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cml.models_v1 as models

# Codeblock 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# Codeblock 20
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=16)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        # self.maxpool
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        # self.maxpool
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        # self.maxpool
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc0 = nn.Linear(in_features=128*6*6, out_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        
    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.maxpool(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)
        
        x = self.dropout(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        
        return x

class Cat_Dog_Dataset():
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        return (image, label)


def load_images(path):

    images = []
    filenames = os.listdir(path)
    
    for filename in tqdm(filenames): 
        if filename == '_DS_Store':
            continue
        image = cv2.imread(os.path.join(path, filename))
        image = cv2.resize(image, dsize=(100,100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    
    return np.array(images)

def show_images(images, labels, start_index):
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20,12))

    counter = start_index

    for i in range(4):
        for j in range(8):
            axes[i,j].set_title(labels[counter].item())
            axes[i,j].imshow(images[counter], cmap='gray')
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)
            counter += 1
    plt.show()
    
def classify_batch(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    device = torch.device('cpu')

    #load the saved model
    model = CNN().to(device)
    model.load_state_dict(torch.load("models/catdog_classifier.pth", map_location=device))

    X_test = np.array(args["X_test"], dtype=np.uint8)
    #print(type(X_test[0][0][0][0]))

    print(X_test.shape, type(X_test)) #Debug
    y_test = np.array(args["y_test"], dtype=np.uint8)
    batch_size = int(args["batch_size"] )
    transforms_test = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    test_dataset  = Cat_Dog_Dataset(images=X_test, labels=y_test, transform=transforms_test)

    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    iter_test = iter(test_loader)
    img_test, lbl_test = next(iter_test)

    # Predict labels
    preds_test = model(img_test.to(device))
    img_test_permuted = img_test.permute(0, 2, 3, 1)
    rounded_preds = preds_test.round().detach().numpy().tolist()
    print(type(rounded_preds))

    # Show test images and the predicted labels .. uncomment when debugging
    #show_images(img_test_permuted, rounded_preds, 0)    
    response = {"result": rounded_preds}
    return response