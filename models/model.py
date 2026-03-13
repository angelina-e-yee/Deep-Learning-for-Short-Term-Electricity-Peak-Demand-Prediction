import torch
import torch.nn as nn

class PrimaryGRU(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, output_size=7):
        super(PrimaryGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # the gru layer: using 12 features from our slimmed down data
        # dropout is at 0.4 so the model doesn't just memorize everything
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True, 
                          dropout=0.4)
        
        # the output head: turning those hidden states into a 7-day forecast
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch, 14, 12]
        # out shape: [batch, 14, 128]
        out, _ = self.gru(x) 
        
        # just grabbing the vibe from the last day of the sequence
        out = out[:, -1, :] 
        
        # mapping that memory to the actual 7-day prediction
        out = self.fc(out)
        return out