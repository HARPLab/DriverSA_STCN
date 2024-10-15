import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd


class GrayscaleTinyImageNet(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if split == 'train':
            train_dir = os.path.join(root, 'train')
            if not os.path.exists(train_dir):
                raise ValueError(f"Train directory not found: {train_dir}")
            for class_dir in os.listdir(train_dir):
                class_path = os.path.join(train_dir, class_dir, 'images')
                if not os.path.exists(class_path):
                    print(f"Warning: Class directory not found: {class_path}")
                    continue
                images = [f for f in os.listdir(class_path) if f.endswith('.JPEG')]
                self.image_paths.extend([os.path.join(class_path, img) for img in images])
                self.labels.extend([class_dir] * len(images))
        else:
            val_dir = os.path.join(root, 'val')
            if not os.path.exists(val_dir):
                raise ValueError(f"Validation directory not found: {val_dir}")
            val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
            if not os.path.exists(val_annotations_file):
                raise ValueError(f"Validation annotations file not found: {val_annotations_file}")
            val_annotations = pd.read_csv(val_annotations_file, sep='\t', header=None)
            self.image_paths = [os.path.join(val_dir, 'images', img) for img in val_annotations[0]]
            self.labels = val_annotations[1].tolist()

        if not self.image_paths:
            raise ValueError(f"No images found for {split} split")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(self.labels)))}
        self.labels = [self.class_to_idx[cls] for cls in self.labels]

        print(f"Loaded {len(self.image_paths)} images for {split} split")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

def modify_resnet18_for_grayscale_tiny_imagenet():
    model = models.resnet18(pretrained=True)
    # Modify the first convolutional layer to accept 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, 200)  # Tiny ImageNet has 200 classes
    return model

# Set up data transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Adjusted for grayscale
])

# Set up datasets and dataloaders
data_path = 'saves/imagenet/tiny-imagenet-200'  # Replace with your Tiny ImageNet data path
train_dataset = GrayscaleTinyImageNet(data_path, split='train', transform=transform)
val_dataset = GrayscaleTinyImageNet(data_path, split='val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize the model, loss function, and optimizer
model = modify_resnet18_for_grayscale_tiny_imagenet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training settings
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Epoch {epoch + 1} Validation Accuracy: {100 * correct / total:.2f}%')

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'resnet18_grayscale_tiny_imagenet.pth')