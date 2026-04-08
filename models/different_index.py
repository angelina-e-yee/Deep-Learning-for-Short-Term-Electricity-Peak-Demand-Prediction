import torch
import matplotlib.pyplot as plt
import numpy as np
from model import PrimaryGRU # Assuming your model is in model.py

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
indices = [100, 200, 300, 400]
titles = ["Index=100", "Index=200", "Index=300", "Index=400"]

# 2. Load Data and Model
x_test, y_test = torch.load("data/tensors/test.tensors.pt")
# Change 128 to 64
model = PrimaryGRU(input_size=12, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load("models/primary_gru_best.pth"))
model.to(device)
model.eval()

# 3. Generate Predictions
with torch.no_grad():
    # Push all test data through to get all preds at once
    all_preds = model(x_test.to(device)).cpu()

# 4. Plotting the Seasonal Stress Test
fig, axes = plt.subplots(len(indices), 1, figsize=(10, 4 * len(indices)))
plt.subplots_adjust(hspace=0.4)

for i, idx in enumerate(indices):
    actual = y_test[idx].numpy()
    pred = all_preds[idx].numpy()
    
    ax = axes[i]
    ax.plot(actual, label='Actual IESO Demand', marker='o', color='#1f77b4', linewidth=2)
    ax.plot(pred, label='GRU Forecast', linestyle='--', marker='x', color='#ff7f0e', linewidth=2)
    
    ax.set_title(f"Seasonal Check: {titles[i]} (Test Index {idx})", fontweight='bold')
    ax.set_ylabel("Scaled Demand")
    ax.set_ylim(0, 1.1) # Keep scale consistent to see magnitude shifts
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

axes[-1].set_xlabel("Days into Future (7-Day Horizon)")

plt.savefig("figures/seasonal_stress_test.png", bbox_inches='tight', dpi=300)
print("Seasonal stress test complete. Check figures/seasonal_stress_test.png")