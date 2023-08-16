import torch
from torch import nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

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

        # Initialize convolutional layers using He Initialization
        #self._init_layers()

    def _init_layers(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

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
        #print(x.size())
        #x = F.relu(F.max_pool2d(self.conv7(x), 2))
        #print(x.size())
        #x = F.relu(F.max_pool2d(self.conv8(x), 2)) 
        #print(x.size())
        #x = F.relu(F.max_pool2d(self.conv9(x), 2))
        #print(x.size())
        x = x.view(-1, 6144) 
        x = self.fully(x)
        #print(x.size())
        return x

# Define your PyTorch model
model = CustomCNN()

sphere_model = CustomCNN()


# Calculate the number of parameters
num_params = count_parameters(model)
print(f"Number of parameters in the model: {num_params}")

num_params = count_parameters(sphere_model)
print(f"Number of parameters in the sphere model: {num_params}")