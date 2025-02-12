# BASIS TRANSFORMATION LAYER ON SPD AND STIEFEL MANIFOLDS

A deep learning framework for EEG classification that leverages manifold-aware techniques. This project implements a basis transformation layer which applies an orthonormal matrix (parameterized on the Stiefel manifold) to transform SPD (symmetric positive definite) matrices. A Log-Euclidean mapping is used to project these matrices onto the tangent space, and a custom optimizer ensures that updates remain on the Stiefel manifold.

## Features

- **Manifold-Aware Layers**
  - **Basis Transformation Layer:** Applies an orthonormal matrix (from the Stiefel manifold) to SPD matrices.
  - **Log-Euclidean Mapping:** Projects SPD matrices onto their tangent space at the identity.
  - **MixOptimizer:** A custom optimizer that reprojects updates to the Stiefel manifold after each gradient step.

- **EEG Data Preprocessing**
  - Data loading, segmentation, band-pass filtering, and mean removal.
  - Covariance matrix computation with regularization to ensure SPD properties.

- **Clean & Modular Codebase**
  - Separated modules for data handling, model definitions, and training routines.
  - Utility functions for manifold-specific linear algebra operations.

## Directory Structure

Below is the directory tree structure for the project:


flowchart TD
    A[eeg_spdnet_project] --> B[data]
    A --> C[models]
    A --> D[utils]
    A --> E[train.py]
    A --> F[requirements.txt]

    B --> B1[__init__.py]
    B --> B2[dataset.py]
    B --> B3[preprocessing.py]

    C --> C1[__init__.py]
    C --> C2[stiefel.py]
    C --> C3[spdnet.py]

    D --> D1[__init__.py]
    D --> D2[math_ops.py]

