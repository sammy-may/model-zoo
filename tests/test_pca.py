import numpy as np

from model_zoo.pca.pca import PCA


def test_pca():
    """
    Unit test for the PCA class.

    Tests that PCA correctly reduces dimensionality and preserves variance
    directions on a small centered dataset.
    """
    # Construct data with clear principal direction
    X = np.array(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2, 1.6],
            [1, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ]
    )

    pca = PCA(n_components=1)
    pca.fit(X)

    # Project and then back-project to approximate original
    X_proj = pca.project(X)
    X_recon = X_proj @ pca.comps.T + pca.mean  # Reconstruct

    # Check shapes
    assert X_proj.shape == (10, 1), "Projection should reduce to 1D"

    # Check variance preserved is high (since it's the first component)
    original_var = np.var(X - np.mean(X, axis=0))
    recon_var = np.var(X_recon - np.mean(X, axis=0))
    assert recon_var / original_var > 0.85, "Most variance should be preserved"
