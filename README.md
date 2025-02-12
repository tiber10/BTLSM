# BASIS TRANSFORMATION LAYER ON SPD AND STIEFEL MANIFOLDS

A deep learning framework for EEG classification that leverages manifold-aware techniques. This project implements a basis transformation layer which applies an orthonormal matrix (parameterized on the Stiefel manifold) to transform SPD (symmetric positive definite) matrices. A Log-Euclidean mapping is used to project these matrices onto the tangent space, and a custom optimizer ensures that updates remain on the Stiefel manifold.

## Features

- **Manifold-Aware Layers**  
  - **Basis Transformation Layer:** Applies an orthonormal matrix (from the Stiefel manifold) to SPD matrices.  
  - **Log-Euclidean Mapping:** Projects SPD matrices onto their tangent space at the identity.  
  - **MixOptimizer:** A custom optimizer that reprojects updates to the Stiefel manifold.

- **EEG Data Preprocessing**  
  - Data loading, segmentation, band-pass filtering, and mean removal.  
  - Covariance matrix computation with regularization to ensure SPD properties.

- **Clean & Modular Codebase**  
  - Clearly separated modules for data handling, model definitions, and training routines.  
  - Utility functions for manifold-specific linear algebra operations.

## Directory Structure

eeg_spdnet_project/ ├── data │   ├── init.py │   ├── dataset.py # Custom PyTorch Dataset for covariance matrices │   └── preprocessing.py # Data loading, segmentation, filtering, and covariance computation ├── models │   ├── init.py │   ├── stiefel.py # StiefelParameter and MixOptimizer definitions │   └── spdnet.py # BasisTransformation, LogEig, and SPDNet definitions ├── utils │   ├── init.py │   └── math_ops.py # Utility functions for SPD and manifold operations ├── train.py # Main training and evaluation script └── requirements.txt # Project dependencies
