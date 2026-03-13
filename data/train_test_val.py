import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import os

# 1. Load your final dataset
df = pd.read_csv("data/processed/daily_weather_and_demand_2002_2025.csv")
df['date'] = pd.to_datetime(df['date'])

# --- CLEANING ---
feature_cols = df.columns.drop('date')
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
df[feature_cols] = df[feature_cols].ffill().fillna(0)

# 2. Chronological Split
train_df = df[df['date'].dt.year <= 2018].copy()
val_df = df[(df['date'].dt.year >= 2019) & (df['date'].dt.year <= 2021)].copy()
test_df = df[df['date'].dt.year >= 2022].copy()

print(f"Train days: {len(train_df)}, Val days: {len(val_df)}, Test days: {len(test_df)}")

# 3. Scaling (Updated to match the slimmed feature list)
# We scale everything except the binary 'is_weekend' 
# (though scaling 0/1 doesn't hurt, it's cleaner to leave it or include it)
features_to_scale = [
    'avg_temperature', 'avg_relative_humidity', 'avg_wind_speed', 
    'avg_hourly_health_index', 'rain', 'snow', 'avg_daily_demand', 
    'year', 'month', 'population_growth', 'day_of_week'
] 

scaler = MinMaxScaler()
train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
val_df[features_to_scale] = scaler.transform(val_df[features_to_scale])
test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

# 4. The Sliding Window Function
def create_sequences(data, sequence_length=14, target_length=7):
    X, y = [], []
    feature_data = data.drop(columns=['date']).values
    
    # Dynamically find index of demand for the target 'y'
    demand_idx = data.columns.get_loc('avg_daily_demand') - 1 
    
    for i in range(len(feature_data) - sequence_length - target_length):
        X.append(feature_data[i : i + sequence_length])
        y.append(feature_data[i + sequence_length : i + sequence_length + target_length, demand_idx])
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

# Create the PyTorch tensors
X_train, y_train = create_sequences(train_df, sequence_length=14, target_length=7)
X_val, y_val = create_sequences(val_df, sequence_length=14, target_length=7)
X_test, y_test = create_sequences(test_df, sequence_length=14, target_length=7)

# Check shapes (Should show 12 features)
print(f"X_train shape: {X_train.shape}") 
print(f"y_train shape: {y_train.shape}") 

# 5. Save the Tensors
os.makedirs("data/tensors", exist_ok=True)
torch.save((X_train, y_train), "data/tensors/train.tensors.pt")
torch.save((X_val, y_val), "data/tensors/val.tensors.pt")
torch.save((X_test, y_test), "data/tensors/test.tensors.pt")

print("\nSuccess! Tensors updated with slimmed feature set.")