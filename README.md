# DL- Developing a Neural Network Classification Model using Transfer Learning


## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Include the problem statement and Dataset


## Neural Network Model
<img width="987" height="792" alt="neural diagram" src="https://github.com/user-attachments/assets/c188cc8c-c2a6-4ae6-8b42-86be64eba25d" />

## DESIGN STEPS
### STEP 1: 

Load the image dataset using ImageFolder and preprocess the images by resizing them to 224 × 224 pixels and converting them into tensors.

### STEP 2: 
Create DataLoader objects for the training and testing datasets with a batch size of 32, using shuffling for the training data.


### STEP 3: 

Load the pre-trained VGG19 model with ImageNet weights and replace its final fully connected layer with a new layer according to the number of classes in the given dataset.



### STEP 4: 

Freeze the VGG19 feature extraction layers to retain the learned ImageNet features, and define Cross-Entropy Loss and the Adam optimizer for training the classifier




### STEP 5: 

Train the modified VGG19 model for multiple epochs by performing forward propagation, calculating the loss, backpropagation, and updating the trainable classifier parameters. Record the training and validation losses.



### STEP 6: 

Evaluate the trained VGG19 model using the test dataset by generating class predictions and calculating the test accuracy. Generate a confusion matrix to visualize the actual and predicted classes and a classification report to obtain precision, recall, and F1-score.


### STEP 7: 


Perform prediction on individual test images using the trained VGG19 model. Display each image with its actual class and predicted class to verify the model's classification performance.


### STEP 8: 

Analyze the obtained training and validation loss, test accuracy, confusion matrix, classification report, and individual image predictions to determine the overall performance of the transfer-learning model.







### Name: Jeyaarikaran P

### Register Number: 212224240064


## PROGRAM
```.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])

!unzip -qq "/content/chip_data (1).zip" -d data

# Load dataset from a folder (structured as: dataset/class_name/images)
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

# Display some input images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()

# Show sample images from the training dataset
show_sample_images(train_dataset)

# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")

# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

# Get the total number of samples in the testing dataset
print(f"Total number of training samples: {len(test_dataset)}")

# Get the shape of the first image in the dataset
first_image1, label = test_dataset[0]
print(f"Shape of the first image : {first_image.shape}")

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

## Step 2: Load Pretrained Model and Modify for Transfer Learning
# Load a pre-trained VGG19 model
from torchvision.models import VGG19_Weights
model = models.vgg19(weights=VGG19_Weights.DEFAULT)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes
num_classes = len(train_dataset.classes)
in_features=model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

summary(model, input_size=(3, 224, 224))

# Freeze all layers except the final layer
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers

# Include the Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: JEYAARIKARAN P")
    print("Register Number:212224240064")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the model
train_model(model, train_loader,test_loader)

## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Name: JEYAARIKARAN P")
    print("Register Number: 212224240064")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Name: JEYAARIKARAN P")
    print("Register Number:212224240064")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Evaluate the model
test_model(model,test_loader)

## Step 5: Predict on a Single Image and Display It
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        # Get the predicted class index
        _, predicted = torch.max(output, 1)
        predicted = predicted.item()


    class_names = class_names = dataset.classes
    # Display the image
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

# Example Prediction
predict_image(model, image_index=55, dataset=test_dataset)

#Example Prediction
predict_image(model, image_index=25, dataset=test_dataset)
```



### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="903" height="619" alt="image" src="https://github.com/user-attachments/assets/754f47c8-c610-4dff-b0b4-126141437fb7" />


## Confusion Matrix :

<img width="1079" height="667" alt="image" src="https://github.com/user-attachments/assets/2286108c-a765-4aaa-bc36-8873ed4353bc" />



## Classification Report


<img width="632" height="264" alt="image" src="https://github.com/user-attachments/assets/aa7b4ca3-f799-486f-afc5-aff5c8d1956e" />


### New Sample Data Prediction :



<img width="601" height="448" alt="image" src="https://github.com/user-attachments/assets/c6ec7354-f97e-4fda-b8af-4e7e0437663d" />





<img width="514" height="447" alt="image" src="https://github.com/user-attachments/assets/7bd11009-e86e-4434-b56f-0016085edd11" />

## RESULT
Thus, an image classification model was developed using transfer learning with the pre-trained VGG19 architecture. The model was trained on the given dataset after preprocessing the images and freezing the VGG19 feature extraction layers. The performance was evaluated using test accuracy, confusion matrix, and classification report with precision, recall, and F1-score. The model was also tested on individual images to compare their actual and predicted classes.
