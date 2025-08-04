import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
    
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=5, stride=2, padding=2),   # (128, 1600)
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),  # (256, 800)
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),  # (512, 400)
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1), # (256, 800)
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1), # (128, 1600)
            nn.ReLU(),
            nn.ConvTranspose1d(128, 1, kernel_size=5, stride=2, padding=2, output_padding=1),  # (80, 3200)
            nn.Tanh()  # or ReLU / Sigmoid depending on your signal
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded