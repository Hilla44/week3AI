"""
MNIST Handwritten Digit Classification using Convolutional Neural Network (CNN)
Dataset: MNIST Handwritten Digits (28x28 grayscale images)

This script demonstrates:
1. Building a CNN architecture for image classification
2. Training the model with proper optimization
3. Achieving >95% test accuracy
4. Visualizing predictions on sample images
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration parameters for the model and training"""
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    learning_rate = 0.001
    momentum = 0.9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_model = True
    model_path = 'mnist_cnn_model.pth'

config = Config()

print("=" * 70)
print("MNIST DIGIT CLASSIFICATION - CNN WITH PYTORCH")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Device: {config.device}")
print(f"  Batch size: {config.batch_size}")
print(f"  Epochs: {config.epochs}")
print(f"  Learning rate: {config.learning_rate}")

# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 1] Loading and Preprocessing MNIST Dataset")
print("=" * 70)

# Define data transformations
# Convert images to tensors and normalize (mean=0.1307, std=0.3081 are MNIST statistics)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
print("\nDownloading and loading MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders for batching and shuffling
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.test_batch_size,
    shuffle=False,
    num_workers=0
)

print(f"\n✓ Dataset loaded successfully!")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Image shape: {train_dataset[0][0].shape}")
print(f"  Number of classes: 10 (digits 0-9)")

# ============================================================================
# STEP 2: CNN MODEL ARCHITECTURE
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 2] Building CNN Model Architecture")
print("=" * 70)

class MNISTConvNet(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification

    Architecture:
    - Conv Layer 1: 1 -> 32 channels, 3x3 kernel
    - Conv Layer 2: 32 -> 64 channels, 3x3 kernel
    - Max Pooling: 2x2
    - Dropout: 25%
    - Fully Connected 1: 9216 -> 128 neurons
    - Dropout: 50%
    - Fully Connected 2: 128 -> 10 neurons (output layer)
    """

    def __init__(self):
        super(MNISTConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Fully connected layers
        # After conv layers and pooling: 28x28 -> 14x14 -> 7x7, with 64 channels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output tensor of shape (batch_size, 10) with class logits
        """
        # First convolutional block
        x = self.conv1(x)           # (batch, 32, 28, 28)
        x = F.relu(x)               # Apply ReLU activation

        # Second convolutional block
        x = self.conv2(x)           # (batch, 64, 28, 28)
        x = F.relu(x)               # Apply ReLU activation
        x = self.pool(x)            # (batch, 64, 14, 14)
        x = self.dropout1(x)        # Apply dropout

        # Flatten for fully connected layers
        x = torch.flatten(x, 1)     # (batch, 64*14*14)

        # First fully connected layer
        x = self.fc1(x)             # (batch, 128)
        x = F.relu(x)               # Apply ReLU activation
        x = self.dropout2(x)        # Apply dropout

        # Output layer
        x = self.fc2(x)             # (batch, 10)

        # Note: We use log_softmax for numerical stability with NLLLoss
        return F.log_softmax(x, dim=1)

# Initialize the model
model = MNISTConvNet().to(config.device)

print("\n✓ CNN Model created successfully!")
print(f"\nModel Architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# STEP 3: LOSS FUNCTION AND OPTIMIZER
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 3] Setting up Loss Function and Optimizer")
print("=" * 70)

# Loss function: Negative Log Likelihood Loss (works with log_softmax)
criterion = nn.NLLLoss()

# Optimizer: Adam optimizer (adaptive learning rate)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Learning rate scheduler: reduce LR when validation accuracy plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)

print(f"✓ Loss function: Negative Log Likelihood Loss")
print(f"✓ Optimizer: Adam (lr={config.learning_rate})")
print(f"✓ Scheduler: ReduceLROnPlateau")

# ============================================================================
# STEP 4: TRAINING FUNCTION
# ============================================================================
def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch

    Args:
        model: The neural network model
        device: Device to train on (CPU or GPU)
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        epoch: Current epoch number

    Returns:
        Average training loss for the epoch
    """
    model.train()  # Set model to training mode
    train_loss = 0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

# ============================================================================
# STEP 5: EVALUATION FUNCTION
# ============================================================================
def evaluate(model, device, test_loader, criterion):
    """
    Evaluate the model on test data

    Args:
        model: The neural network model
        device: Device to evaluate on (CPU or GPU)
        test_loader: DataLoader for test data
        criterion: Loss function

    Returns:
        Average test loss and accuracy
    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():  # Disable gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Calculate loss
            test_loss += criterion(output, target).item()

            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy

# ============================================================================
# STEP 6: TRAINING LOOP
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 6] Training the CNN Model")
print("=" * 70)

# Track training history
history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

print("\nStarting training...")
start_time = time.time()

for epoch in range(1, config.epochs + 1):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{config.epochs}")
    print(f"{'='*70}")

    # Train for one epoch
    train_loss, train_acc = train(model, config.device, train_loader, optimizer, criterion, epoch)

    # Evaluate on test set
    test_loss, test_acc = evaluate(model, config.device, test_loader, criterion)

    # Update learning rate scheduler
    scheduler.step(test_acc)

    # Save metrics
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)

    # Print epoch summary
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

    # Early stopping if we achieve >98% accuracy
    if test_acc > 98.0:
        print(f"\n✓ Achieved {test_acc:.2f}% accuracy! Stopping early.")
        break

training_time = time.time() - start_time

print(f"\n{'='*70}")
print("Training Completed!")
print(f"{'='*70}")
print(f"Total training time: {training_time:.2f} seconds")
print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")

# Save the model
if config.save_model:
    torch.save(model.state_dict(), config.model_path)
    print(f"\n✓ Model saved to {config.model_path}")

# ============================================================================
# STEP 7: VISUALIZE TRAINING HISTORY
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 7] Visualizing Training History")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
ax1.plot(history['train_loss'], label='Train Loss', marker='o')
ax1.plot(history['test_loss'], label='Test Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True)

# Plot accuracy
ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
ax2.plot(history['test_acc'], label='Test Accuracy', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True)
ax2.axhline(y=95, color='r', linestyle='--', label='95% Target')

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✓ Training history saved to 'training_history.png'")
plt.close()

# ============================================================================
# STEP 8: VISUALIZE PREDICTIONS ON SAMPLE IMAGES
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 8] Visualizing Predictions on Sample Images")
print("=" * 70)

def visualize_predictions(model, device, test_loader, num_samples=5):
    """
    Visualize model predictions on sample images

    Args:
        model: The trained model
        device: Device to run inference on
        test_loader: DataLoader for test data
        num_samples: Number of samples to visualize
    """
    model.eval()

    # Get a batch of test images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # Select random samples
    indices = np.random.choice(len(images), num_samples, replace=False)
    sample_images = images[indices]
    sample_labels = labels[indices]

    # Make predictions
    with torch.no_grad():
        sample_images = sample_images.to(device)
        outputs = model(sample_images)
        probabilities = torch.exp(outputs)  # Convert log probabilities to probabilities
        predictions = outputs.argmax(dim=1)

    # Move back to CPU for visualization
    sample_images = sample_images.cpu()
    predictions = predictions.cpu()
    probabilities = probabilities.cpu()

    # Create visualization
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for idx in range(num_samples):
        ax = axes[idx]

        # Display image
        img = sample_images[idx].squeeze()
        ax.imshow(img, cmap='gray')

        # Get prediction and confidence
        pred = predictions[idx].item()
        true_label = sample_labels[idx].item()
        confidence = probabilities[idx][pred].item() * 100

        # Set title with prediction
        color = 'green' if pred == true_label else 'red'
        ax.set_title(
            f'Pred: {pred}\nTrue: {true_label}\nConf: {confidence:.1f}%',
            color=color,
            fontsize=10
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Sample predictions saved to 'sample_predictions.png'")
    plt.close()

    # Print detailed predictions
    print("\nDetailed Predictions:")
    print("=" * 70)
    for idx in range(num_samples):
        pred = predictions[idx].item()
        true_label = sample_labels[idx].item()
        confidence = probabilities[idx][pred].item() * 100

        # Get top 3 predictions
        top3_prob, top3_pred = torch.topk(probabilities[idx], 3)

        print(f"\nSample {idx + 1}:")
        print(f"  True Label: {true_label}")
        print(f"  Predicted: {pred} (Confidence: {confidence:.2f}%)")
        print(f"  Result: {'✓ CORRECT' if pred == true_label else '✗ INCORRECT'}")
        print(f"  Top 3 predictions:")
        for i, (p, prob) in enumerate(zip(top3_pred, top3_prob)):
            print(f"    {i+1}. Digit {p.item()}: {prob.item()*100:.2f}%")

# Visualize predictions
visualize_predictions(model, config.device, test_loader, num_samples=5)

# ============================================================================
# STEP 9: CONFUSION MATRIX
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 9] Generating Confusion Matrix")
print("=" * 70)

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_confusion_matrix(model, device, test_loader):
    """Generate and plot confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - MNIST CNN')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to 'confusion_matrix.png'")
    plt.close()

    # Classification report
    print("\nClassification Report:")
    print("=" * 70)
    print(classification_report(all_labels, all_preds,
                                target_names=[str(i) for i in range(10)]))

plot_confusion_matrix(model, config.device, test_loader)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

final_accuracy = history['test_acc'][-1]
target_achieved = "✓ YES" if final_accuracy >= 95.0 else "✗ NO"

print(f"""
Model Performance:
  - Final Test Accuracy: {final_accuracy:.2f}%
  - Target (>95%) Achieved: {target_achieved}
  - Training Time: {training_time:.2f} seconds
  - Total Epochs: {len(history['test_acc'])}
  - Total Parameters: {total_params:,}

Model Architecture:
  - 2 Convolutional Layers (32 and 64 filters)
  - Max Pooling (2x2)
  - 2 Fully Connected Layers (128 and 10 neurons)
  - Dropout Regularization (25% and 50%)
  - ReLU Activation Functions

Files Generated:
  ✓ mnist_cnn_model.pth - Trained model weights
  ✓ training_history.png - Training/test loss and accuracy plots
  ✓ sample_predictions.png - Predictions on 5 sample images
  ✓ confusion_matrix.png - Confusion matrix visualization

The CNN successfully classifies MNIST handwritten digits with high accuracy!
""")
print("=" * 70)
