import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch.nn.functional as F

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


data_transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor()])

# Load the full CSV file
df = pd.read_csv('/home/mstveras/structured3d_repo/img_classes.csv')

# Split the DataFrame into train and test sets (you can modify this accordingly)
train_df = df[:1000]
test_df = df[1000:]

# Create custom datasets and data loaders
train_dataset = CustomDataset(train_df, "/home/mstveras/structured3d_repo/struct3d_images", transform=data_transform)
test_dataset = CustomDataset(test_df, "/home/mstveras/structured3d_repo/struct3d_images", transform=data_transform)

# Define batch size and set shuffle to True (if needed)
batch_size = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define the model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32 * 64, 512),  # Update this to match the output size from the convolutional layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 40),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)  # Print the shape of the tensor after convolutional layers
        x = self.fc_layers(x)
        return x  


# Create an instance of the model
model = CustomCNN()


# Move the model to the GPU if available
if torch.cuda.is_available():
    model = model.cuda()

# Define the optimizer and loss function
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training parameters
epochs = 10

'''
# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode

    running_loss = 0.0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader):
        # Move data to the GPU if available
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item()
        total_samples += inputs.size(0)

    # Print training statistics for each epoch
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / total_samples:.4f}")
'''

model.load_state_dict(torch.load("trained_model_weights.pth"))
#torch.save(model.state_dict(), "trained_model_weights.pth")

def evaluate_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device
    test_loss = 0
    test_total_samples = 0

    with torch.no_grad():
        all_predictions = []
        all_labels = []

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            #print(outputs)

            # Convert the sigmoid outputs to binary predictions
            predictions = (outputs).float()

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            print(all_predictions)

            # Calculate test loss
            test_loss += criterion(outputs, labels).item()
            test_total_samples += inputs.size(0)

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)  

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='micro')
    recall = recall_score(all_labels, all_predictions, average='micro')

    return accuracy, precision, recall

# Assuming you have already trained the model and have the test_loader
accuracy, precision, recall = evaluate_model(model, test_loader)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")