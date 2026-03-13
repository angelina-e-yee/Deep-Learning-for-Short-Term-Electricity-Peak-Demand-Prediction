import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Setup Folders
os.makedirs("figures", exist_ok=True)

# 2. Load the New 12-Feature Tensors
X_train, y_train = torch.load("data/tensors/train.tensors.pt")
X_val, y_val = torch.load("data/tensors/val.tensors.pt")

# Set up DataLoaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

# 3. Define the Baseline Architecture (Updated for 12 features)
class LinearBaseline(nn.Module):
    # num_features is now 12 to match our slimmed dataset
    def __init__(self, seq_length=14, num_features=12, output_days=7): 
        super().__init__()
        self.flatten = nn.Flatten()
        # 14 days * 12 features = 168 input nodes mapping to 7 output days
        self.linear = nn.Linear(seq_length * num_features, output_days)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

# 4. Training Setup
model = LinearBaseline()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
train_losses, val_losses = [], []

print("Training Slimmed Linear Baseline (12 Features)...")

for epoch in range(epochs):
    model.train()
    batch_train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        batch_train_loss += loss.item()
        
    avg_train_loss = batch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    batch_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            batch_val_loss += loss.item()
            
    avg_val_loss = batch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

# 5. Save Learning Curve
plt.figure(figsize=(8, 5))
skip = 2 
plt.plot(range(skip, epochs), train_losses[skip:], label='Training Loss (MSE)')
plt.plot(range(skip, epochs), val_losses[skip:], label='Validation Loss (MSE)')
plt.title('Baseline Model Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('figures/baseline_learning_curve_12features.png')

# 6. Sample Prediction (Index 100 for April 2022 consistency)
model.eval()
with torch.no_grad():
    sample_X = X_val[100].unsqueeze(0)
    sample_y_true = y_val[100].numpy()
    sample_y_pred = model(sample_X).squeeze().numpy()

plt.figure(figsize=(8, 5))
plt.plot(sample_y_true, label='Actual Demand', marker='o')
plt.plot(sample_y_pred, label='Baseline Prediction', marker='x', linestyle='--')
plt.title('Baseline Qualitative Result (Index 100)')
plt.xlabel('Day')
plt.ylabel('Scaled Demand')
plt.legend()
plt.savefig('figures/baseline_sample_prediction.png')

print("Baseline training complete. Figures saved.")