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
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init

from spherenet import SphereConv2D, SphereMaxPool2D
from torch import nn
import torchvision.datasets as datasets
from equi_conv import EquiConv2d, equi_conv2d
torch.cuda.empty_cache()

torch.cuda.empty_cache()

import torchvision.models as models

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
        #self.conv9 = SphereConv2D(3*256, 3*512, stride=1)
        #self.pool9 = SphereMaxPool2D(stride=2)
        self.fully = nn.Sequential(nn.Linear(6144,3072), nn.Linear(3072, 10))


    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = F.relu(self.pool5(self.conv5(x)))
        x = F.relu(self.pool6(self.conv6(x)))
        #x = F.relu(self.pool7(self.conv7(x)))
        #x = F.relu(self.pool8(self.conv8(x)))
        #print(x.size())
        #x = F.relu(self.pool9(self.conv9(x)))
        x = x.view(-1, 6144)  # flatten, [B, C, H, W) -> (B, C*H*W)
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
    correct_predictions = 0

    with torch.no_grad():
        all_predictions = []
        all_labels = []

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Convert the sigmoid outputs to binary predictions
            predictions = (torch.sigmoid(outputs)> 0.5).float()

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            #correct_predictions += torch.sum(predictions == labels).item()

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            total_samples += inputs.size(0)

    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)

    test_loss = running_loss/total_samples

    #precision = precision_score(all_labels, all_predictions, average='micro')
    #recall = recall_score(all_labels, all_predictions, average='micro')
    #f1 = f1_score(all_labels, all_predictions, average='micro')
    print(f'Test loss = {test_loss}')
    #print(f'precision = {precision}, recall = {recall}, f1-score= {f1_score}')
    return test_loss

# Training loop
def train(model, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    #for epoch in range(epochs):
    model.train()  # Set the model to training mode

    running_loss = 0.0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_samples += inputs.size(0)

    train_loss = running_loss / total_samples
    print(f"Epoch [{epoch + 1}],\n Train Loss: {train_loss:.4f}")
    
    return train_loss
    

#torch.save(model.state_dict(), "trained_model_weights_equi.pth")
#model.load_state_dict(torch.load("trained_model_weights.pth"))


def main():

    # Create an instance of the model
    #model = CustomVGG16(num_classes=11)
    sphere_model = SphereNet()
    #model_cnn = CustomCNN()
    #sphere_model.load_state_dict(torch.load("best_spher_model.pth"))

    if torch.cuda.is_available():
        sphere_model = sphere_model.cuda()
        #model_cnn = model_cnn.cuda()


    #scheduler,

    #optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=1e-4)
    optimizer_sphere = torch.optim.Adam(sphere_model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 500
    #torch.manual_seed(args.seed)

    #device = torch.device('cuda')

    data_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()])

    # Load the full CSV file
    df = pd.read_csv('/home/mstveras/pesquisa-mestrado/img_classes_big_edited.csv')

    # Split the DataFrame into train and test sets (you can modify this accordingly)
    train_df = df[:3435]
    test_df = df[3435:4171]

    # Create custom datasets and data loaders
    train_dataset = CustomDataset(train_df, "/home/mstveras/struct3d-data", transform=data_transform)
    test_dataset = CustomDataset(test_df, "/home/mstveras/struct3d-data", transform=data_transform)

    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_train=[]
    loss_train_std=[]
    loss_test=[]
    loss_test_std=[]
    best_val_loss = float('inf')
    patience = 10

    scheduler = StepLR(optimizer_sphere, step_size=20, gamma=0.1)  # Adjust the step_size and gamma values as needed

    for epoch in range(1, epochs + 1):
        
        ## SphereCNN
        print('{} Sphere CNN {}'.format('='*10, '='*10))
        loss_train.append(train(sphere_model, train_loader, optimizer_sphere, epoch))
        val_loss = test(sphere_model, test_loader)
        loss_test.append(val_loss)

        # Conventional CNN
        #print('{} Conventional CNN {}'.format('='*10, '='*10))
        #loss_train_std.append(train(model_cnn, train_loader, optimizer_cnn, epoch))
        #val_loss = test(model_cnn, test_loader)
        #loss_test_std.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(sphere_model.state_dict(), 'best_spher_model_lccr3.pth')  # Save the best model
            #torch.save(model_cnn.state_dict(), 'best_sphere_model.pth')  # Save the best model

        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

        # Conventional CNN
        #print('{} Conventional CNN {}'.format('='*10, '='*10))
        #loss_train_std.append(train(model_cnn, train_loader, optimizer_cnn))
        #loss_test_std.append(test(model_cnn, test_loader))


    #informar o vetor de Ã©pocas no lugar no np.array(list.........
    epochs = np.array(list(range(0,2*len(loss_train),2)))


    plt.figure(figsize=(10, 5))

    plt.plot(epochs, loss_train, 'r', label='Training Loss') # plotting t, a separately 
    plt.plot(epochs, loss_test, 'b', label='Validation Loss') # plotting t, b separately 
    plt.plot(epochs, loss_train_std, 'g', label='Training Loss std') # plotting t, a separately 
    plt.plot(epochs, loss_test_std, 'y', label='Validation Loss std') # plotting t, b separately 
    plt.legend(loc='upper right')
    plt.grid(linestyle=':')
    plt.xlabel('Epochs',fontsize=18)
    plt.ylabel('Loss',fontsize=18)

    plt.show()



if __name__ == '__main__':
    main()
