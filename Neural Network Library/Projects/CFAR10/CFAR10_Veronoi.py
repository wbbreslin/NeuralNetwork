import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import ssl

# Custom Voronoi layer
class VoronoiLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(VoronoiLayer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(num_classes, in_features))

    def forward(self, x):
        x = x.unsqueeze(1) - self.centroids.unsqueeze(0)
        x = torch.norm(x, dim=2)  # Euclidean distance
        x = torch.argmin(x, dim=1)  # Assign each sample to the closest centroid
        return x

# Define a CNN model with Voronoi classifier
class CNNVoronoi(nn.Module):
    def __init__(self, num_classes):
        super(CNNVoronoi, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.voronoi = VoronoiLayer(64 * 8 * 8, num_classes)  # Custom Voronoi layer

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.voronoi(x)
        return x

class CustomVoronoiLoss(nn.Module):
    def __init__(self):
        super(CustomVoronoiLoss, self).__init__()

    def forward(self, predicted, targets, centroids):
        # Calculate the Euclidean distance between predicted centroids and true centroids
        distances = torch.norm(centroids[predicted] - centroids[targets], dim=1)
        loss = torch.mean(distances)
        return loss

# Bypass SSL verification by setting the context
ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset with SSL bypassed
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = CNNVoronoi(num_classes=10)  # Assuming 10 classes in CIFAR-10
criterion = CustomVoronoiLoss()  # Using the custom loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Calculate centroids for true labels (this assumes labels are indices of the centroids)
        true_centroids = model.voronoi.centroids[labels]

        # Calculate custom loss using the predicted centroids, true labels, and true centroids
        loss = criterion(outputs, labels, true_centroids)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")