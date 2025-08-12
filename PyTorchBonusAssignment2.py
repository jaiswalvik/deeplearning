# Step 1: Import Necessary Libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Step 2: Set manual seed
torch.manual_seed(42)

# Step 3: Define Transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5),  # Normalize each channel
                         (0.5, 0.5, 0.5))
])

# Step 4: Download and Load CIFAR-10 Dataset

# Training data
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=True  # Shuffle during training
)

print(f"Number of images in the training dataset: {len(train_dataset)}")
image, label = train_dataset[0]
print(f"Shape of images in the training dataset: {image.shape}")
# Test data
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False  # No shuffle during testing
)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layer 1: input channels=3, output channels=6, kernel size=5
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        # Max pooling: kernel size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Convolutional layer 2: input channels=6, output channels=16, kernel size=5
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # After conv and pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))   # Conv2 -> ReLU -> Pool
        x = x.view(-1, 16 * 5 * 5)             # Flatten
        x = F.relu(self.fc1(x))                # FC1 -> ReLU
        x = F.relu(self.fc2(x))                # FC2 -> ReLU
        x = self.fc3(x)                        # FC3 -> Output (no activation)
        return x
    
net = Net()
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total_params}")

trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Loss function: Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# Optimizer: SGD with momentum 0.9, learning rate 0.01
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Get one batch of training data
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Forward pass
outputs = net(images)

# Compute loss
loss = criterion(outputs, labels)

print(f"Initial loss value: {loss.item():.4f}")

# Training loop for 2 epochs
for epoch in range(2):  # loop over the dataset twice
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = net(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backpropagation
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/2], Average Loss: {avg_loss:.4f}")

print("Finished Training")

correct = 0
total = 0

# Disable gradient calculation for testing
with torch.no_grad():
    for images, labels in test_loader:
        # Forward pass
        outputs = net(images)
        
        # Get predicted class with highest score
        _, predicted = torch.max(outputs, 1)
        
        # Update counts
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy in range [0, 1]
accuracy = correct / total
print(f"Accuracy on the test dataset: {accuracy:.4f}")
