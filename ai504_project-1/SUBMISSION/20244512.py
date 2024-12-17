import torch
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the CNN model
class AI504model01(nn.Module):
    def __init__(self):
        super(AI504model01, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 1 * 1, 512)  # Adjusted to the correct size after pooling
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(torch.relu(self.batch_norm4(self.conv4(x))))

        # Flatten the tensor dynamically
        x = torch.flatten(x, 1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Load the Fashion-MNIST dataset
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True )
    test_loader  = DataLoader(test_set , batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Training the model
def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Step the scheduler to adjust the learning rate
        scheduler.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')


## Save / Evaluation tools

# Generate logits for the test set
def generate_logits(model, test_loader, device):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            logits_list.append(outputs.cpu().numpy())

    logits = np.vstack(logits_list)
    return logits

# Save the logits to a .npy file
def save_logits(logits, file_name):
    np.save(file_name, logits)

# calculate accuracy in runtime
def calc_accuracy(model, test_loader):
    model.eval()
    correct = 0.0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).float().sum().item()
    print(f'Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')

## Main Script

set_seed()

student_id = "20244512"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, test_loader = load_data(batch_size=64)

# Initialize model, loss function, and optimizer
model = AI504model01().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Add learning rate scheduler (StepLR reduces LR by gamma every step_size epochs)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Train the model (with scheduler)
train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=20)

# Generate logits for the test set
logits = generate_logits(model, test_loader, device)

# Save the logits as a .npy file
save_logits(logits, f"{student_id}.npy")

# Calculate accuracy of the final model
calc_accuracy(model, test_loader)