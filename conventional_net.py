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
torch.cuda.empty_cache()
from sklearn.metrics import hamming_loss, jaccard_score, f1_score

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
        #print(image)


        return image, labels


data_transform = transforms.Compose([
    transforms.Resize((256, 512)),
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
batch_size = 3

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



class CustomCNN2(nn.Module):
    def __init__(self):
        super(CustomCNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512 * 8 * 16, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 40)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(F.max_pool2d(self.conv8(x), 2))
        x = F.relu(self.conv9(x))
        x = F.relu(F.max_pool2d(x, 2))
        
        x = x.view(-1, 512 * 8 * 16) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = nn.Sigmoid(x)

        return x



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
        self.fully = nn.Sequential(nn.Linear(3072, 1536), nn.Dropout(0.2), nn.Linear(1536, 40))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2)) 
        x = F.relu(F.max_pool2d(self.conv6(x), 2))
        x = F.relu(F.max_pool2d(self.conv7(x), 2))
        #x = F.relu(F.max_pool2d(self.conv8(x), 2)) 
        #x = F.relu(F.max_pool2d(self.conv9(x), 2))
        
        #print(x.size())
        x = x.view(-1, 3072) 
        x = self.fully(x)
        #print(x.size())
        return x


# Create an instance of the model
model = CustomCNN()


# Move the model to the GPU if available
if torch.cuda.is_available():
    model = model.cuda()

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
criterion = nn.BCEWithLogitsLoss()

# Training parameters
epochs = 30

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
        #outputs = torch.sigmoid(outputs)
        #print(torch.sigmoid(outputs))

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


torch.save(model.state_dict(), "trained_model_weights.pth")

#model.load_state_dict(torch.load("trained_model_weights.pth"))

def evaluate_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device
    test_loss = 0
    test_total_samples = 0
    correct_predictions = 0

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
            predictions = (torch.sigmoid(outputs)> 0.5).float()

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            #print(all_predictions)
            correct_predictions += torch.sum(predictions == labels).item()

            # Calculate test loss
            test_loss += criterion(outputs, labels).item()
            test_total_samples += inputs.size(0)

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        accuracy = correct_predictions / test_total_samples  

        #jaccard_index = jaccard_score(all_labels, all_predictions, average='micro')
    #accuracy = accuracy_score(all_labels, all_predictions)
    #%precision = precision_score(all_labels, all_predictions, average='micro')
    #recall = recall_score(all_labels, all_predictions, average='micro')

    return accuracy

# Assuming you have already trained the model and have the test_loader
accuracy = evaluate_model(model, test_loader)

print(f"Test Accuracy: {accuracy:.4f}")
#print(f"Test Precision: {precision:.4f}")
#print(f"Test Recall: {recall:.4f}")
