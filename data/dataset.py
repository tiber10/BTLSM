# data/dataset.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class SPDDataset(Dataset):
    def __init__(self, csv_file, desired_classes, matrix_shape=None):
        """
        Args:
            csv_file (str): Path to the CSV file.
            desired_classes (list): List of desired class labels to include.
            matrix_shape (tuple, optional): (n_channels, n_samples) shape to reshape each trial's raw data.
                If provided, the raw data will be reshaped and a covariance matrix computed.
                If None, assumes the CSV already contains flattened covariance matrices that can be reshaped to (n, n).
        """
        self.dataframe = pd.read_csv(csv_file)
        
        self.dataframe = self.dataframe[self.dataframe['label'].isin(desired_classes)]
        
        self.labels = self.dataframe['label'].values
        
        data_columns = [col for col in self.dataframe.columns if col not in ['trial', 'label']]
        self.raw_data = self.dataframe[data_columns].values.astype(np.float32)
        self.matrix_shape = matrix_shape
        
        if matrix_shape is not None:
            self.cov_matrices = [self.compute_covariance(trial.reshape(matrix_shape))
                                 for trial in self.raw_data]
        else:
            # Attempt to reshape it to a square matrix.
            n = int(np.sqrt(self.raw_data.shape[1]))
            self.cov_matrices = [trial.reshape((n, n)) for trial in self.raw_data]

    def compute_covariance(self, trial):
        """
        Compute the covariance matrix for a trial.
        Assumes trial is a 2D numpy array with shape (n_channels, n_samples).
        """
        # Subtract the mean (across samples) for each channel
        trial = trial - np.mean(trial, axis=1, keepdims=True)
        # Compute the covariance matrix
        cov = np.cov(trial)
        # Regularize to ensure positive definiteness
        cov += 1e-5 * np.eye(cov.shape[0])
        return cov

    def __len__(self):
        return len(self.cov_matrices)

    def __getitem__(self, idx):
        cov_matrix = torch.from_numpy(self.cov_matrices[idx]).float()
        label = self.labels[idx]
        return cov_matrix, label
