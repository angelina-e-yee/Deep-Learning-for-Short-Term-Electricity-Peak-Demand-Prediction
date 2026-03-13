# Short-Term Electricity Demand Prediction for Ontario

**APS360: Applied Fundamentals of Machine Learning**

This repository contains a deep learning pipeline to forecast the next 7 days of electricity demand in Ontario. By integrating historical load data with weather variables and population growth metrics, the project implements a Stacked Gated Recurrent Unit (GRU) network to capture complex temporal dependencies over a 23-year period (2002-2025).

## Execution Flow

### 1. Pre-processing

Clean the raw data and generate the chronological tensor splits:

```bash
python data_processing.py
python train_test_val.py

```

### 2. Training and Validation

Train the primary GRU model. This script monitors validation loss and saves the best weights to models/primary_gru_best.pth.

```bash
python train.py

```

### 3. Final Evaluation

Assess the model on the held-out test set (2022-2025) to generate quantitative metrics:

```bash
python test.py

```

---

## Model Architecture

* Input: 14-day history (14 x 16 features).
* Core: 2-Layer Stacked GRU (64 hidden units).
* Regularization: 0.2 Dropout.
* Output: Fully Connected head mapping to 7-day forecast.

---

## Results

* Baseline: Linear Regression served as a temporal-agnostic floor.
* Primary: The GRU successfully smoothed validation spikes and captured seasonal momentum.
* Test Set Performance: [Insert Final Test MSE Here]

---

## Author

Angelina Yee
University of TorontoFaculty of Applied Science and Engineering