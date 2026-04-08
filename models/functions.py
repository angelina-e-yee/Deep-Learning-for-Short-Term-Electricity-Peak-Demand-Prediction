import torch
import torch.nn as nn

class AsymmetricMSELoss(nn.Module):
    def __init__(self, penalty_factor=1.3):
        super(AsymmetricMSELoss, self).__init__()
        # We store the penalty factor here when we initialize
        self.penalty_factor = penalty_factor

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        
        # If error is positive (y_true > y_pred -> we under-predicted the peak)
        # apply the penalty multiplier before squaring.
        squared_error = torch.where(error > 0, (error * self.penalty_factor)**2, error**2)
        
        return squared_error.mean()
    

import torch
import torch.nn as nn

class PeakWeightedMSELoss(nn.Module):
    def __init__(self, peak_weight=3.0):
        super(PeakWeightedMSELoss, self).__init__()
        # How much more we care about errors on high-demand days
        self.peak_weight = peak_weight

    def forward(self, y_pred, y_true):
        # Calculate standard squared error
        squared_error = (y_true - y_pred)**2
        
        # Create a dynamic weight based on the actual demand.
        # If y_true is 0, weight is 1 (normal MSE). 
        # If y_true is large, the weight scales up.
        dynamic_weights = 1.0 + (y_true * self.peak_weight)
        
        # Apply the weights to the error
        weighted_error = squared_error * dynamic_weights
        
        return weighted_error.mean()