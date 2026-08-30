import numpy as np

from scripts.pipeline.compute_ood_stats import (
    _apply_pca,
    _fit_pca,
    _l2_normalize,
    _mahalanobis_distances,
    _mahalanobis_to_mean,
    _regularize_covariance,
)


def test_l2_normalize_produces_unit_norm_vectors():
    features = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    normalized = _l2_normalize(features)

    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0])
    np.testing.assert_allclose(normalized[0], [0.6, 0.8, 0.0])


def test_regularize_covariance_adds_ridge_proportional_to_trace():
    covariance = np.diag([2.0, 4.0])
    regularized = _regularize_covariance(covariance, ridge_scale=0.5)

    # trace=6, feature_dim=2 -> ridge = 0.5 * (6/2) = 1.5
    np.testing.assert_allclose(regularized, np.diag([3.5, 5.5]))


def test_regularize_covariance_makes_singular_matrix_invertible():
    # Covarianza degenerada (rango 1): sin regularizar, pinv no equivale a inv real.
    covariance = np.array([[1.0, 1.0], [1.0, 1.0]])
    regularized = _regularize_covariance(covariance, ridge_scale=0.1)

    # No debe lanzar y el resultado debe ser simetrico positivo-definido (autovalores > 0).
    eigenvalues = np.linalg.eigvalsh(regularized)
    assert np.all(eigenvalues > 0)


def test_mahalanobis_to_mean_matches_squared_euclidean_with_identity_covariance():
    features = np.array([[3.0, 0.0], [0.0, 4.0]])
    mean = np.array([0.0, 0.0])
    identity = np.eye(2)

    distances = _mahalanobis_to_mean(features, mean, identity)

    np.testing.assert_allclose(distances, [9.0, 16.0])


def test_fit_pca_retains_only_components_up_to_explained_variance():
    # Solo las primeras 3 dimensiones cargan varianza real; el resto es ruido casi nulo.
    rng = np.random.default_rng(0)
    real_signal = rng.normal(size=(500, 3)) * np.array([10.0, 5.0, 2.0])
    noise = rng.normal(scale=1e-6, size=(500, 47))
    features = np.concatenate([real_signal, noise], axis=1)

    mean, components, explained = _fit_pca(features, explained_variance=0.99)

    assert mean.shape == (50,)
    assert components.shape[1] == 50
    # No debe quedarse con las 50 dimensiones: el ruido no aporta varianza real.
    assert components.shape[0] < 10
    assert explained >= 0.99


def test_apply_pca_centers_and_projects():
    features = np.array([[5.0, 7.0, 9.0]])
    pca_mean = np.array([1.0, 2.0, 3.0])
    # Componentes = ejes 0 y 1: se queda con las dos primeras coordenadas tras centrar.
    pca_components = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    projected = _apply_pca(features, pca_mean, pca_components)

    np.testing.assert_allclose(projected, [[4.0, 5.0]])


def test_pca_reduction_removes_noise_floor_domination_from_inverse_covariance():
    # Regresion del bug real: sin reducir dimensionalidad antes de regularizar, un
    # feature vector con muchas dimensiones de ruido casi-nulo deja la covarianza
    # regularizada con autovalores de piso (~ridge) en la mayoria de las dimensiones,
    # y esas dimensiones de piso dominan la energia de la inversa (ver
    # docs/es/deep-learning/ood-detection.md). Tras reducir con PCA, el numero de
    # autovalores de piso cae a ~0 y ya no dominan.
    rng = np.random.default_rng(2)
    real_signal = rng.normal(size=(2000, 5))
    noise = rng.normal(scale=1e-4, size=(2000, 195))
    features = _l2_normalize(np.concatenate([real_signal, noise], axis=1))

    def floored_energy_fraction(covariance: np.ndarray) -> float:
        regularized = _regularize_covariance(covariance)
        ridge = 1e-3 * np.trace(covariance) / covariance.shape[0]
        eigenvalues = np.linalg.eigvalsh(regularized)
        inv_eigenvalues = 1.0 / eigenvalues
        floored = eigenvalues < 2 * ridge
        return inv_eigenvalues[floored].sum() / inv_eigenvalues.sum()

    raw_covariance = np.cov(features, rowvar=False)
    raw_fraction = floored_energy_fraction(raw_covariance)

    _, components, _ = _fit_pca(features, explained_variance=0.99)
    reduced = _apply_pca(features, features.mean(axis=0), components)
    reduced_covariance = np.cov(reduced, rowvar=False)
    reduced_fraction = floored_energy_fraction(reduced_covariance)

    # Sin PCA, la mayoria de la energia inversa viene del piso de ruido.
    assert raw_fraction > 0.4
    # Con PCA, ese piso deja de dominar.
    assert reduced_fraction < 0.1


def test_rmd_cancels_dimensions_shared_between_class_and_background_models():
    # Dimension 0 discrimina: su media de clase difiere de la media de fondo.
    # Dimensiones 1 y 2 comparten exactamente la misma media y la misma covarianza
    # entre el modelo de clase y el de fondo, como pasaria si no llevan señal
    # relacionada con la clase (el caso que Ren et al. 2021 identifican como el
    # que hace fallar a la distancia de Mahalanobis "plana" en near-OOD).
    class_mean = np.array([5.0, 100.0, -50.0])
    background_mean = np.array([0.0, 100.0, -50.0])  # dims 1,2 iguales a las de clase
    inv_covariance = np.diag([1.0, 0.01, 0.02])  # misma matriz para clase y fondo

    # Punto de prueba cerca del centroide de clase en la dimension 0 (la unica que
    # discrimina), pero lejos en las dimensiones "ruido" (1 y 2).
    feature = np.array([[5.5, 9999.0, -8888.0]])

    class_distance = _mahalanobis_distances(
        feature, np.array([0]), class_mean[None, :], inv_covariance
    )[0]
    background_distance = _mahalanobis_to_mean(feature, background_mean, inv_covariance)[0]
    rmd_score = class_distance - background_distance

    # Sin RMD, la distancia de Mahalanobis plana queda dominada por el ruido de las
    # dimensiones 1 y 2 (miles), aunque el punto esta pegado al centroide en la
    # dimension que sí importa.
    assert class_distance > 1000
    # Con RMD, esas dimensiones se cancelan exactamente y el score depende solo de
    # la dimension 0.
    expected_rmd = ((5.5 - 5.0) ** 2 * 1.0) - ((5.5 - 0.0) ** 2 * 1.0)
    np.testing.assert_allclose(rmd_score, expected_rmd)
