# models/spdnet.py
import torch
from torch import nn
import geotorch
from models.stiefel import StiefelParameter  # Import from the stiefel module

class BasisTransformation(nn.Module):
    def __init__(self, n_channels):
        super(BasisTransformation, self).__init__()
        self.n_channels = n_channels
        # Initialize W as an identity matrix
        W = torch.eye(n_channels)
        self.W = nn.Parameter(W)
        # Enforce orthogonality via GeoTorch
        geotorch.orthogonal(self, 'W')

    def forward(self, X):
        # X shape: (batch_size, channels, channels)
        W = self.W
        return W.t() @ X @ W

class LogEig(nn.Module):
    def forward(self, X):
        # X shape: (batch_size, channels, channels)
        eigenvalues, eigenvectors = torch.linalg.eigh(X)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        log_eigenvalues = torch.log(eigenvalues)
        return eigenvectors @ torch.diag_embed(log_eigenvalues) @ eigenvectors.transpose(-1, -2)

class SPDNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SPDNet, self).__init__()
        self.basis = BasisTransformation(n_channels)
        self.logeig = LogEig()
        self.flatten = nn.Flatten()
        # Flatten the upper triangular part of the matrix. For an SPD matrix,
        # the number of unique elements is n_channels*(n_channels+1)//2.
        self.output_size = (n_channels * (n_channels + 1)) // 2
        self.fc = nn.Sequential(
            nn.Linear(self.output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, X):
        # Apply basis transformation and LogEig mapping
        X = self.basis(X)
        X = self.logeig(X)
        # Extract the upper triangular part
        indices = torch.triu_indices(X.size(1), X.size(2), device=X.device)
        X = X[:, indices[0], indices[1]]
        return self.fc(X)
