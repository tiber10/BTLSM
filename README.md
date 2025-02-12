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


## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/basis_transformation_layer_on_spd_and_stiefel_manifolds.git
   cd basis_transformation_layer_on_spd_and_stiefel_manifolds
2. **Set Up a Virtual Environment (Recommended):**
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

4. **Install Dependencies:**
   pip install -r requirements.txt
