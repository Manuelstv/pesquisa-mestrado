import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from torchviz import make_dot

class CustomDataset(Dataset):
    def __init__(self, dataframe, directory, transform=None):
        self.dataframe = dataframe
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.dataframe.iloc[idx, 1]) # Assuming 'combined' contains the image file names
        image = Image.open(img_name).convert('RGB')
        labels = torch.tensor(self.dataframe.iloc[idx, 2:].values.astype(np.float32))  # Convert labels to float32


        if self.transform:
            image = self.transform(image)

        return image, labels

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 3*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3*2, 3*4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3*4, 3*8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(3*8, 3*16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(3*16, 3*32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(3*32, 3*64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(3*64, 3*128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(3*128, 3*256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(3*256, 3*512, kernel_size=3, padding=1)
        self.fully = nn.Sequential(nn.Linear(3072, 10))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2)) 
        x = F.relu(F.max_pool2d(self.conv6(x), 2))
        x = F.relu(F.max_pool2d(self.conv7(x), 2))
        x = F.relu(F.max_pool2d(self.conv8(x), 2)) 
        x = F.relu(F.max_pool2d(self.conv9(x), 2))
        
        x = x.view(-1, 3072) 
        x = self.fully(x)
        return x

# Create an instance of your model
model = CustomCNN()

data_transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor()])

# Load the full CSV file
df = pd.read_csv('/home/mstveras/pesquisa-mestrado/img_classes_big_edited.csv')

# Split the DataFrame into train and test sets (you can modify this accordingly)
train_df = df[:10]
# Create custom datasets and data loaders
train_dataset = CustomDataset(train_df, "/home/mstveras/struct3d-data", transform=data_transform)

import os
from PIL import Image
import numpy as np

batch_size = 2

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for i, (inputs, labels) in enumerate(train_loader):
    # Create the visualization figure
    dot = make_dot(model(inputs), params=dict(model.named_parameters()))

selected_layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5,
                   model.conv6, model.conv7, model.conv8, model.conv9, model.fully]

# Create the visualization figure for the selected layers
dot = make_dot(model(inputs), params=dict(model.named_parameters()))

# Save the figure
dot.format = 'png'  # Set the output format (other formats are available)
dot.render("custom_cnn_graph")
