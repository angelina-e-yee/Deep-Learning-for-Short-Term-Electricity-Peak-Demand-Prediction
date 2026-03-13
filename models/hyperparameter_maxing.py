import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import PrimaryGRU 
import pandas as pd
import os

# 1. getting ready
# making a folder so the models have somewhere to live
os.makedirs("models/grid_search", exist_ok=True)
x_train, y_train = torch.load("data/tensors/train.tensors.pt")
x_val, y_val = torch.load("data/tensors/val.tensors.pt")

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

# 2. the search party
# picking which combos we actually want to test
lrs = [0.0005, 0.001, 0.01]
hidden_sizes = [32, 64, 128]
results = []

print("starting manual grid search...")

for h_size in hidden_sizes:
    for lr in lrs:
        print(f"\n--- testing: hidden={h_size}, lr={lr} ---")
        
        # waking up the model
        model = PrimaryGRU(input_size=12, hidden_size=h_size, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_for_run = float('inf')
        
        # 30 epochs is enough to see if it's actually learning anything
        for epoch in range(30):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            
            model.eval()
            v_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    v_loss += criterion(model(xb), yb).item()
            
            avg_val_loss = v_loss / len(val_loader)
            if avg_val_loss < best_val_for_run:
                best_val_for_run = avg_val_loss
        
        print(f"finished run. best val mse: {best_val_for_run:.6f}")
        results.append({'hidden_size': h_size, 'lr': lr, 'best_val_mse': best_val_for_run})

# 4. the results table
results_df = pd.DataFrame(results)
results_df.to_csv("grid_search_results.csv", index=False)
print("\nsuccess! results saved to grid_search_results.csv")