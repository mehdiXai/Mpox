import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from tqdm import tqdm
import sys 

# Set the output encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# ========================
# 1️⃣ Loading the Dataset with Data Augmentation
# ========================
data_dir = "C:/Users/HP pro/OneDrive/Desktop/memoir/teste_belhia/Augimg (2)/Augimg"  # Path containing both class folders
num_images = 204  # Number of images to use

# 📌 Transformations with Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 📌 Load dataset
train_full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transform)
eval_full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=eval_transform)
if len(train_full_dataset.classes) != 2:
    raise ValueError(
        f"Expected exactly 2 class folders in {data_dir}, found {train_full_dataset.classes}"
    )
num_images = min(num_images, len(train_full_dataset))
generator = torch.Generator().manual_seed(42)
subset_dataset, _ = random_split(
    train_full_dataset,
    [num_images, len(train_full_dataset) - num_images],
    generator=generator
)

# 📌 Split into training and test sets
train_size = int(0.83 * num_images)
test_size = num_images - train_size
train_dataset, test_dataset = random_split(
    subset_dataset,
    [train_size, test_size],
    generator=generator
)
test_indices = [subset_dataset.indices[i] for i in test_dataset.indices]
train_indices = [subset_dataset.indices[i] for i in train_dataset.indices]
train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
test_dataset = torch.utils.data.Subset(eval_full_dataset, test_indices)

# 📌 DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ========================
# 2️⃣ Define MobileNetV3 Model (Large Version)
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)  # Adjust for 2 classes
model = model.to(device)

# ========================
# 3️⃣ Training with Early Stopping and Learning Rate Scheduling
# ========================
num_epochs = 150 
early_stopping_patience = 10
best_loss = float("inf")
epochs_no_improve = 0

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 📌 Label Smoothing
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 📌 Reduced initial LR
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    with tqdm(train_loader, desc=f"📌 Epoch {epoch+1}/{num_epochs}") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 📌 Evaluate on the test set
    model.eval()
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"📌 Epoch [{epoch+1}/{num_epochs}] - Loss (Train/Test): {train_loss:.4f}/{test_loss:.4f}, Accuracy (Train/Test): {train_acc:.2f}%/{test_acc:.2f}%")

    scheduler.step(test_loss)

    # 📌 Early Stopping
    if test_loss < best_loss:
        best_loss = test_loss
        epochs_no_improve = 0
        best_model = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.state_dict().items()
        }
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(f"📌 Early stopping triggered after {epoch+1} epochs without improvement.")
        break

# 📌 Load the best model
model.load_state_dict(best_model)

# ==============================
# 4️⃣ Optimizing Classification Threshold
# ==============================
y_true, y_pred, y_scores = [], [], []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]

        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

# 📌 Find the best threshold
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
valid_precision_recall = precision[:-1] * recall[:-1]
optimal_idx = np.argmax(valid_precision_recall)
optimal_threshold = thresholds[optimal_idx]
print(f"📌 Best classification threshold: {optimal_threshold:.2f}")

y_pred = [1 if score >= optimal_threshold else 0 for score in y_scores]

# ==============================
# 5️⃣ Plot Confusion Matrix & ROC Curve
# ==============================
conf_matrix = confusion_matrix(y_true, y_pred)
class_names = ["Monkey pox", "Others"]  # Replace with actual class names

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# 📌 Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="magma", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# 📌 ROC Curve Plot
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# ==============================
# 📌 6️⃣ Loss & Accuracy Evaluation with Recognition Rate
# ==============================

epochs = range(1, len(train_losses) + 1)

# 📌 Display the final recognition rate
final_test_acc = test_accs[-1]
print(f"📌 Final Recognition Rate: {final_test_acc:.2f}%")

plt.figure(figsize=(12, 5))

# 📌 Loss Curve (Train vs. Test)
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Training Loss", marker='o', color='blue')
plt.plot(epochs, test_losses, label="Test Loss", marker='o', color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Evaluation")
plt.legend()

# 📌 Accuracy Curve (Train vs. Test) + Recognition Rate
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label="Training Accuracy", marker='o', color='blue')
plt.plot(epochs, test_accs, label="Test Accuracy", marker='o', color='red')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Evaluation with Recognition Rate")
plt.legend()

plt.show()
