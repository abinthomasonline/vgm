import dask.array as da
import numpy as np
from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky


def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    if covariance_type != "full":
        raise NotImplementedError("Only full covariance type is implemented")

    resp = resp.persist()

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = da.dot(resp.T, X) / da.expand_dims(nk, axis=1)
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        # "tied": _estimate_gaussian_covariances_tied,
        # "diag": _estimate_gaussian_covariances_diag,
        # "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return da.compute(nk, means, covariances)


def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = []
    for k in range(n_components):
        diff = X - means[k]
        cov_k = da.dot(resp[:, k] * diff.T, diff) / nk[k]
        cov_k = cov_k + reg_covar * da.eye(n_features)
        covariances.append(cov_k)
    return da.stack(covariances)


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision
    # matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

    if covariance_type == "full":
        log_prob = []
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = da.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob.append(da.sum(da.square(y), axis=1))
        log_prob = da.stack(log_prob, axis=1)

    elif covariance_type == "tied":
        raise NotImplementedError("Tied covariance type is not implemented")

    elif covariance_type == "diag":
        raise NotImplementedError("Diagonal covariance type is not implemented")

    elif covariance_type == "spherical":
        raise NotImplementedError("Spherical covariance type is not implemented")

    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
