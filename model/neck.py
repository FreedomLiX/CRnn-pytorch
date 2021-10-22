
import torch
import torch.nn as nn


# Neck, nn.LSTM(), encode
class EncoderWithRNN(nn.Module):
    """
    encoder with LSTM .
    input shape : B W C .
    output shape : B W hidden_size * 2 if bidirectional=True, otherwise , B W hidden_size * 1
    """

    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, num_layers=2, bidirectional=True)

    def forward(self, x):
        assert (len(list(x.shape)) == 3)
        x, _ = self.lstm(x)
        return x


class SequenceEncoder(nn.Module):
    """
    Encode Backbone' outputs as : used RNN(LSTM)
    B, C, H, W ==> (B W hidden_size* (1 or 2))
    """

    def __init__(self, in_channels, hidden_size=48):
        super(SequenceEncoder, self).__init__()
        self.in_channels = in_channels
        self.encoder = EncoderWithRNN(self.in_channels, hidden_size=hidden_size)

    def forward(self, x):
        # -1, 2048, 1, 25 ==> -1, 1, 25,2048
        B, C, H, W = x.shape
        assert H == 1
        x = x.permute(0, 2, 3, 1)
        # -1, 1, 25,2048 ==> -1, 25, 2048 (B W C)
        x = x.squeeze(1)
        # -1, 25, 2048 ==> -1, 25, hidden_size* (1 or 2)
        x = self.encoder(x)
        return x
