import torch
from model import PrimaryGRU
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. grabbing the data
# using the 2022-2025 stuff the model has literally never seen before
x_test, y_test = torch.load("data/tensors/test.tensors.pt")

# 2. waking up the model
# making sure it knows we have 12 features to deal with
model = PrimaryGRU(input_size=12, hidden_size=64, num_layers=2)

# 3. loading the winner
# pulling in the best weights from that 50-epoch grind
model.load_state_dict(torch.load("models/primary_gru_best.pth"))
model.eval()

# 4. checking the math
# seeing how much the model actually messes up
criterion = torch.nn.MSELoss()
with torch.no_grad():
    preds = model(x_test)
    test_loss = criterion(preds, y_test)

print("-" * 30)
print(f"final test mse (2022-2025): {test_loss.item():.6f}")
print("-" * 30)

# 5. checking the vibes: picking a random sample
# day 100 should be somewhere deep into the test set
idx = 100 
plt.figure(figsize=(10, 5))
plt.plot(y_test[idx].numpy(), label='actual demand (ieso)', marker='o')
plt.plot(preds[idx].numpy(), label='gru prediction', linestyle='--', marker='x')

plt.title(f"out-of-sample test prediction (index {idx})")
plt.xlabel("days into the future (forecast horizon)")
plt.ylabel("scaled demand")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figures/primary_test_sample.png")

print("saved qualitative test plot to figures/primary_test_sample.png")