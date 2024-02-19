import torch.nn as nn
import torch
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMVehicleTracker(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, seq_length=30):
        super(LSTMVehicleTracker, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 12)  # Output is now 6 pairs of 2D (X,Y)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(
            device
        )
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(
            device
        )

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn_last_layer = hn[-1, :, :]  # Get the hidden state of the last layer
        out = self.fc(hn_last_layer)
        return out.view(-1, 6, 2)  # Reshape the output to have 6 pairs of (X,Y)
