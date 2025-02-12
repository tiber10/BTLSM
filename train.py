# train.py
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from models.spdnet import SPDNet
from models.stiefel import MixOptimizer
from data.dataset import SPDDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Data Loading & Preprocessing
# -------------------------------
# Specify desired classes and the CSV file path (update with your actual CSV file)
desired_classes = ['left_hand', 'feet', 'tongue']
csv_file = 'data/bciiv2a.csv'


matrix_shape = None


dataset = SPDDataset(csv_file=csv_file, desired_classes=desired_classes, matrix_shape=matrix_shape)

# Train-validation split using indices (stratified by the dataset labels)
indices = np.arange(len(dataset))
train_indices, val_indices = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=dataset.labels
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# Model Initialization
# -------------------------------
# Extract the number of channels from the dataset (using matrix_shape)
n_channels = dataset.matrix_shape[0]
n_classes = len(desired_classes)
model = SPDNet(n_channels, n_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
optimizer = MixOptimizer(optimizer)  # Wrap the optimizer for manifold constraints

# -------------------------------
# Training & Evaluation Functions
# -------------------------------
def testNetwork(net, loader, criterion, device):
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = net(xb)
            loss = criterion(outputs, yb)
            running_loss += loss.item() * yb.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    print(f'Test Loss: {running_loss/total:.4f} | Test Acc: {correct/total:.4f}')
    print('Confusion Matrix:\n', confusion_matrix(all_labels, all_preds))
    print('Classification Report:\n', classification_report(all_labels, all_preds, target_names=desired_classes))
    return correct/total

def trainNetwork(net, trainloader, validloader, model_path='best_spdnet_model', epochs=150):
    best_loss = float('inf')
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = net(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        net.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for xb, yb in validloader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = net(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * yb.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == yb).sum().item()
                total_val += yb.size(0)
        val_loss /= total_val
        val_acc = correct_val / total_val

        print(f"Epoch {epoch+1}/{epochs} -- Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_file = os.path.join(model_path, 'best_model.pt')
            torch.save(net.state_dict(), model_file)
            print("Saved best model.")

    # Load best model for final evaluation
    net.load_state_dict(torch.load(model_file))
    return net

# -------------------------------
# Run Training
# -------------------------------
if __name__ == '__main__':
    trained_model = trainNetwork(model, train_loader, val_loader, epochs=150)
    print("Final evaluation on validation set:")
    testNetwork(trained_model, val_loader, criterion, device)
