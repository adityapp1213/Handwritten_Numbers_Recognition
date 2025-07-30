# digit_recognizer.py

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
epochs = 3  # You may increase this to improve accuracy further
lr = 0.001

# Transform to convert PIL images to tensors and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset (downloads automatically if not present)
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Define a simple Feedforward Neural Network for digit classification
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)   # First hidden layer
        self.fc2 = nn.Linear(128, 128)     # Second hidden layer
        self.fc3 = nn.Linear(128, 10)      # Output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28*28)              # Flatten 28x28 image to 784 vector
        x = F.relu(self.fc1(x))            # Apply ReLU activation
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)  # Output log-probabilities
        return x

# Initialize model, loss function, and optimizer
model = DigitClassifier().to(device)
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
print("Training the model...")
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for batch in train_loader:
        data, target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f" Epoch {epoch+1}/{epochs} complete. Average Loss: {total_loss / len(train_loader):.4f}")

# Evaluation on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
accuracy = correct / total * 100
print(f" Test Accuracy: {accuracy:.2f}%")

# Ensure accuracy is high enough
if accuracy < 97:
    print("Accuracy below 97%. Consider increasing epochs or tuning hyperparameters.")

# Save the trained model to disk
torch.save(model.state_dict(), "digit_classifier.pth")
print("Model saved as 'digit_classifier.pth'")

# Custom image prediction from digits/digitX.png
image_number = 1
print("\n Starting predictions on custom digit images...")
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))  # Resize to 28x28 if not already
        img = np.invert(img) / 255.0     # Invert image and normalize to [0, 1]
        img_tensor = torch.tensor(img, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, 28, 28]

        with torch.no_grad():
            output = model(img_tensor)
            predicted_digit = output.argmax(dim=1).item()
            print(f" Image {image_number}: Predicted digit → {predicted_digit}")
            plt.imshow(img, cmap="gray")
            plt.title(f"Predicted: {predicted_digit}")
            plt.axis("off")
            plt.show()
        image_number += 1
    except Exception as e:
        print(f"Error reading image {image_number}: {e}")
        image_number += 1
