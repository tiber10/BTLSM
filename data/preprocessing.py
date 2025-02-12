# data/preprocessing.py
import numpy as np
from sklearn.preprocessing import LabelEncoder

def segment_trials(X, y, segment_length_s=1, sampling_rate=250):
    """
    Segments each trial into segments of fixed length.
    
    Args:
        X (np.ndarray): Data with shape (num_trials, num_channels, num_samples).
        y (np.ndarray): Labels for each trial.
        segment_length_s (int): Segment length in seconds.
        sampling_rate (int): Sampling rate in Hz.
    
    Returns:
        (np.ndarray, np.ndarray): Segmented data and corresponding labels.
    """
    num_trials, num_channels, num_samples = X.shape
    samples_per_segment = segment_length_s * sampling_rate
    segments_per_trial = num_samples // samples_per_segment

    X_segments, y_segments = [], []
    for i in range(num_trials):
        for j in range(segments_per_trial):
            start = j * samples_per_segment
            end = start + samples_per_segment
            segment = X[i, :, start:end]
            X_segments.append(segment)
            y_segments.append(y[i])
    return np.array(X_segments), np.array(y_segments)

def compute_covariance_matrices(epochs):
    """
    Computes covariance matrices for each epoch.
    
    Args:
        epochs (np.ndarray): EEG epochs with shape (num_epochs, num_channels, num_samples).
    
    Returns:
        np.ndarray: Covariance matrices of shape (num_epochs, num_channels, num_channels).
    """
    cov_matrices = []
    for epoch in epochs:
        # Mean removal
        epoch = epoch - np.mean(epoch, axis=1, keepdims=True)
        cov = np.cov(epoch)
        # Regularization for positive definiteness
        cov += 1e-5 * np.eye(cov.shape[0])
        cov_matrices.append(cov)
    return np.array(cov_matrices)

def encode_labels(y):
    """
    Encodes string labels into integers.
    
    Returns:
        (np.ndarray, LabelEncoder): Encoded labels and the fitted encoder.
    """
    le = LabelEncoder()
    labels_encoded = le.fit_transform(y)
    return labels_encoded, le
