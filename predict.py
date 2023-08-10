import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss, jaccard_score, f1_score
import torch.nn.functional as F

import torch.nn.init as init

from spherenet import SphereConv2D, SphereMaxPool2D
from torch import nn
import torchvision.datasets as datasets
from equi_conv import EquiConv2d, equi_conv2d
torch.cuda.empty_cache()

torch.cuda.empty_cache()

import torchvision.models as models

def calculate_zero_one_accuracy(predictions, labels):
    all_correct = np.all(predictions == labels, axis=1)
    zero_one_accuracy = np.mean(all_correct)
    return zero_one_accuracy

class SphereNet(nn.Module):
    def __init__(self):
        super(SphereNet, self).__init__()
        self.conv1 = SphereConv2D(3, 3*2, stride=1)
        self.pool1 = SphereMaxPool2D(stride=2)
        self.conv2 = SphereConv2D(3*2, 3*4, stride=1)
        self.pool2 = SphereMaxPool2D(stride=2)
        self.conv3 = SphereConv2D(3*4, 3*8, stride=1)
        self.pool3 = SphereMaxPool2D(stride=2)
        self.conv4 = SphereConv2D(3*8, 3*16, stride=1)
        self.pool4 = SphereMaxPool2D(stride=2)
        self.conv5 = SphereConv2D(3*16, 3*32, stride=1)
        self.pool5 = SphereMaxPool2D(stride=2)
        self.conv6 = SphereConv2D(3*32, 3*64, stride=1)
        self.pool6 = SphereMaxPool2D(stride=2)
        self.conv7 = SphereConv2D(3*64, 3*128, stride=1)
        self.pool7 = SphereMaxPool2D(stride=2)
        self.conv8 = SphereConv2D(3*128, 3*256, stride=1)
        self.pool8 = SphereMaxPool2D(stride=2)
        self.conv9 = SphereConv2D(3*256, 3*512, stride=1)
        self.pool9 = SphereMaxPool2D(stride=2)
        self.fully = nn.Sequential(nn.Linear(3072, 10))

        # Initialize convolutional layers using He Initialization
        self._init_layers()

    def _init_layers(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = F.relu(self.pool5(self.conv5(x)))
        x = F.relu(self.pool6(self.conv6(x)))
        x = F.relu(self.pool7(self.conv7(x)))
        x = F.relu(self.pool8(self.conv8(x)))
        x = F.relu(self.pool9(self.conv9(x)))
        x = x.view(-1, 3072)  # flatten, [B, C, H, W) -> (B, C*H*W)
        x = self.fully(x)        
        return x

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

def test(model, data_loader):
    criterion = nn.BCEWithLogitsLoss()
    model.eval() 
    device = next(model.parameters()).device
    running_loss = 0
    total_samples = 0

    with torch.no_grad():
        all_predictions = []
        all_labels = []

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Convert the sigmoid outputs to binary predictions
            predictions = (torch.sigmoid(outputs) > 0.5).float()

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            total_samples += inputs.size(0)

    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)

    test_loss = running_loss / total_samples
    hamming_loss_value = hamming_loss(all_labels, all_predictions)
    zero_one_accuracy = calculate_zero_one_accuracy(all_predictions, all_labels)

    print(f'Test loss = {test_loss}')
    print(f'Hamming loss = {hamming_loss_value}')
    print(f'Zero-One Accuracy = {zero_one_accuracy}')

    return test_loss, hamming_loss_value

# Define the model
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
        #self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.fc = nn.Linear(8192, 4096)
        self.fully = nn.Sequential(nn.Linear(3072, 10))

        # Initialize convolutional layers using He Initialization
        self._init_layers()

    def _init_layers(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

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
        #print(x.size())
        return x


#torch.save(model.state_dict(), "trained_model_weights_equi.pth")
#model.load_state_dict(torch.load("trained_model_weights.pth"))


def main():

    # Create an instance of the model
    #model = CustomVGG16(num_classes=11)
    sphere_model = SphereNet()
    model_cnn = CustomCNN()

    
    model_cnn.load_state_dict(torch.load("best_planar_model.pth"))

    sphere_model.load_state_dict(torch.load("best_spher_model.pth"))

    if torch.cuda.is_available():
        sphere_model = sphere_model.cuda()
        model_cnn = model_cnn.cuda()
        
    #scheduler,

    data_transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor()])

    # Load the full CSV file
    df = pd.read_csv('/home/mstveras/pesquisa-mestrado/img_classes_big_edited.csv')

    test_df = df[4171:]

    test_dataset = CustomDataset(test_df, "/home/mstveras/struct3d-data", transform=data_transform)

    batch_size = 2

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    print('=======Conventional CNN ========')
    val_loss = test(model_cnn, test_loader)
    print('=======Spherical CNN ===========')

    val_loss = test(sphere_model, test_loader)



if __name__ == '__main__':
    main()
