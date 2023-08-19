import argparse
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

import torchvision
import torchvision.datasets as datasets
from equi_conv import EquiConv2d, equi_conv2d
torch.cuda.empty_cache()

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import torchvision.models as models

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 3*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3*2, 3*4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3*4, 3*8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(3*8, 3*16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(3*16, 3*32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(3*32, 3*64, kernel_size=3, padding=1)
        #self.conv7 = nn.Conv2d(3*64, 3*128, kernel_size=3, padding=1)
        #self.conv8 = nn.Conv2d(3*128, 3*256, kernel_size=3, padding=1)
        #s#elf.conv9 = nn.Conv2d(3*256, 3*512, kernel_size=3, padding=1)
        #self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.fc = nn.Linear(8192, 4096)
        self.fully = nn.Sequential(nn.Linear(6144,3072), nn.Linear(3072, 10))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #print(x.size())
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #print(x.size())
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        #print(x.size())
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        #print(x.size())
        x = F.relu(F.max_pool2d(self.conv5(x), 2)) 
        #print(x.size())
        x = F.relu(F.max_pool2d(self.conv6(x), 2))
        x = x.view(-1, 6144) 
        x = self.fully(x)
        #print(x.size())
        return x


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
        #self.conv7 = SphereConv2D(3*64, 3*128, stride=1)
        #self.pool7 = SphereMaxPool2D(stride=2)
        #self.conv8 = SphereConv2D(3*128, 3*256, stride=1)
        #self.pool8 = SphereMaxPool2D(stride=2)
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
            #print(inputs)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_samples += inputs.size(0)
    
    writer.flush()

    train_loss = running_loss / total_samples
    print(f"Epoch [{epoch + 1}],\n Train Loss: {train_loss:.4f}")
    
    return train_loss
    

#torch.save(model.state_dict(), "trained_model_weights_equi.pth")
#model.load_state_dict(torch.load("trained_model_weights.pth"))

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def main():

    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer, options={"adam, sgd"}')
    parser.add_argument('--mode', type=str, default='train',
                        help='train/test, options={"train,test"}')
    parser.add_argument('--net', type=str, default='sph',
                        help='con/sph, options={"con, sph"}')
    parser.add_argument('--lr', type=float, default=1E-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before saving model weights')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    
    # Create an instance of the model
    #model = CustomVGG16(num_classes=11)
    if args.net == 'sph':
        print('{} Sphere CNN {}'.format('='*10, '='*10))
        model = SphereNet()
    
    if args.net == 'con':
        print('{} Conventional CNN {}'.format('='*10, '='*10))
        model = CustomCNN()

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
    
    criterion = nn.BCEWithLogitsLoss()

    epochs = args.epochs

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

    batch_size = args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('example_imgs', img_grid)

    loss_train=[]
    loss_test=[]
    best_val_loss = float('inf')
    patience = 10

    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)  # Adjust the step_size and gamma values as needed

    for epoch in range(1, epochs + 1):
        
        ## SphereCNN
        loss_train.append(train(model, train_loader, optimizer, epoch))
        val_loss = test(model, test_loader)
        loss_test.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_sphere_tb2.pth')  # Save the best model
            #torch.save(model_cnn.state_dict(), 'best_sphere_model.pth')  # Save the best model
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

        #accuracy = accuracy_score(all_labels, all_predictions)
        #precision = precision_score(all_labels, all_predictions, average='micro')
        #recall = recall_score(all_labels, all_predictions, average='micro')
        #f1 = f1_score(all_labels, all_predictions, average='micro')

        ##writer.add_scalar("Metrics/accuracy", accuracy, epoch)
        #writer.add_scalar("Metrics/precision", precision, epoch)
        #writer.add_scalar("Metrics/recall", recall, epoch)
        #writer.add_scalar("Metrics/f1_score", f1_score, epoch)

        # Log parameter histograms
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        # Log notes and hyperparameters
        writer.add_text("Notes", "Experiment results: SphereNet model training.")
        hyperparams = {"learning_rate": 1e-4, "batch_size": batch_size, "epochs": epochs}
        #writer.add_hparams(hyperparams, {"hparam/loss": loss})

    writer.close()
    

    # Custom plot for loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(loss_train, label="Train Loss")
    plt.plot(loss_test, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    writer.add_figure("Loss Curves", plt.gcf(), global_step=epoch)

    writer.close()
    
if __name__ == '__main__':
    main()
