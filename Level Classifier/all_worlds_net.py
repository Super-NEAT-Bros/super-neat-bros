#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
from sklearn.cluster import KMeans

import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd         
from torch import Tensor                  
import torch.nn as nn                     
import torch.nn.functional as F           
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[2]:


def get_level_image_list():    
    images = []
    for f in glob.iglob("images/?-?.png"):
        images.append(f)
    images.sort()
    return images


# In[3]:


def get_image_arrays(image_list):    
    images_arrays = []
    for f in image_list:
        np_image = np.array(Image.open(f).convert('RGB')).astype(float)
        images_arrays.append(np_image)
        print(f, np_image.shape)
    return images_arrays


# In[4]:


def create_level_slices():
    image_set_array = np.array([])
    target_set_array = np.array([])
    stride = 5
    for i, image in enumerate(get_image_arrays(get_level_image_list())):
        windowed_image = image[:, (np.arange((image.shape[1] - 240) // stride) * stride)[:, np.newaxis] + np.arange(240)].transpose(1, 0, 2, 3)
        image_set_array = np.vstack((image_set_array, windowed_image)) if image_set_array.size else windowed_image
        target_set_array = np.concatenate((target_set_array, np.full((windowed_image.shape[0],), i))) if target_set_array.size else np.full((windowed_image.shape[0],), i)
        print(windowed_image.shape)
    return torch.from_numpy(image_set_array.transpose(0, 3, 1, 2)).type(torch.FloatTensor), torch.from_numpy(target_set_array).type(torch.FloatTensor)


# In[5]:


image_tensor, target_tensor = create_level_slices()


# In[6]:


class TransformTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


# In[7]:


level_dataset = TransformTensorDataset((image_tensor, target_tensor), transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))


# In[8]:


batch_size = 128
val_split = .1
test_split = .1
shuffle_dataset = True


# In[9]:


dataset_size = len(level_dataset)
indices = list(range(dataset_size))
test_split_i = int(np.floor(test_split * dataset_size))
val_split_i = int(np.floor((test_split + val_split) * dataset_size))
if shuffle_dataset:
    #np.random.seed(35)
    np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[val_split_i:], indices[test_split_i:val_split_i], indices[:test_split_i]

train_loader = torch.utils.data.DataLoader(level_dataset, batch_size=batch_size, 
                                           sampler=SubsetRandomSampler(train_indices))

val_loader = torch.utils.data.DataLoader(level_dataset, batch_size=batch_size, 
                                           sampler=SubsetRandomSampler(val_indices))

test_loader = torch.utils.data.DataLoader(level_dataset, batch_size=batch_size,
                                                sampler=SubsetRandomSampler(test_indices)
)


# In[28]:


import torch.nn as nn 

class LevelClassifier(nn.Module):   
    def __init__(self):
        super(LevelClassifier, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 64, kernel_size=12, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(256, 484, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(484),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(484, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.linear_layers(x)
        return x


# In[29]:


model = LevelClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# In[30]:


PRINT_FREQ = 2


for epoch in range(15):
    model.train()
    
    total_running_loss = 0.0
    running_loss = 0.0
    print(f"Starting Epoch: {epoch}...")
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_running_loss += loss.item()
        if (i + 1) % PRINT_FREQ == 0:    # print every PRINT_FREQ batches
            print(f"Epoch: {epoch} | Average train loss: {running_loss / PRINT_FREQ} | ({batch_size * (i + 1)} / {len(train_loader.dataset)})")
            running_loss = 0.0
    
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
                  
    print(f"""Finished Epoch: {epoch} | Average train loss: {total_running_loss / (len(train_loader.dataset) / batch_size)} | ({len(train_loader.dataset)}/{len(train_loader.dataset)}) | Validation Accuracy: {100 * val_correct / val_total}
""")


# In[31]:


test_correct = 0
test_total = 0
model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * test_correct / test_total))


# In[6]:


torch.save(model.state_dict(), "model.pt")


# In[ ]:




