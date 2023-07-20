import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

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
        #labels = torch.tensor(self.dataframe.iloc[idx, 2:].values, dtype=torch.float32)
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


def visualize_data(data_loader, num_images=4):
    # Get an iterator for the data loader
    data_iter = iter(data_loader)

    # Get the next batch of data
    images, labels = next(data_iter)

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=num_images, ncols=1, figsize=(8, 12))

    for i in range(num_images):
        image = images[i].permute(1, 2, 0)  # Convert tensor to (H, W, C) format for imshow
        label = labels[i]

        axes[i].imshow(image)
        axes[i].set_title(f"Labels: {label}")

    plt.tight_layout()
    plt.show()

# Example usage:
#visualize_data(train_loader, num_images=4)


import torch
import torch.nn as nn

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
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 40),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Create an instance of the model
model = CustomCNN()

# Define the optimizer and loss function
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Move the model to the GPU if available
if torch.cuda.is_available():
    model = model.cuda()

# Assuming you have already defined the model, optimizer, and criterion as shown in the previous code.

# Training parameters
epochs = 10

# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    
    running_loss = 0.0
    correct_predictions = 0
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

