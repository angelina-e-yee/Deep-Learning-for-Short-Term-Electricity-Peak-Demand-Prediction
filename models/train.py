import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import PrimaryGRU 
import matplotlib.pyplot as plt
import os

# 1. making space for our stuff
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# 2. loading the tensors (now with 12 features)
x_train, y_train = torch.load("data/tensors/train.tensors.pt")
x_val, y_val = torch.load("data/tensors/val.tensors.pt")

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

# 3. waking up the model with 12 inputs
model = PrimaryGRU(input_size=12, hidden_size=64, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. the main event: training loop
# doing 50 epochs because the 0.4 dropout needs time to cook
epochs = 50 
train_losses, val_losses = [], []
best_val_loss = float('inf')

print(f"starting training on {len(x_train)} samples...")

for epoch in range(epochs):
    model.train()
    t_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
    
    model.eval()
    v_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            v_loss += criterion(model(xb), yb).item()
            
    avg_train_loss = t_loss / len(train_loader)
    avg_val_loss = v_loss / len(val_loader)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    # keeping the winner
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "models/primary_gru_best.pth")
        status = "(new best!)"
    else:
        status = ""

    print(f"epoch {epoch+1:02d}: train={avg_train_loss:.6f} | val={avg_val_loss:.6f} {status}")

# 5. plotting the drama (the learning curve)
plt.figure(figsize=(10, 5))
# skipping the first 2 epochs so we can actually see the curve zoom
plt.plot(train_losses[2:], label='train loss') 
plt.plot(val_losses[2:], label='val loss')
plt.title("primary gru learning curve (12 features, 0.4 dropout)")
plt.xlabel("epoch")
plt.ylabel("mse")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("figures/primary_learning_curve.png")

print("\ntraining complete. best model saved to models/primary_gru_best.pth")