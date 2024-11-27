import numbers
import warnings

import dask.array as da
import numpy as np
from scipy.special import betaln, digamma, logsumexp
from sklearn.base import _fit_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture._base import _check_shape
from sklearn.mixture._bayesian_mixture import (_log_dirichlet_norm,
                                               _log_wishart_norm)
from sklearn.mixture._gaussian_mixture import (_check_precision_matrix,
                                               _check_precision_positivity,
                                               _compute_log_det_cholesky)
from sklearn.utils import check_array, check_random_state
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted

from .utils import _estimate_gaussian_parameters, _estimate_log_gaussian_prob


class DBGM(BayesianGaussianMixture):
    """Distributed Implementation of Bayesian Gaussian Mixture Model"""

    _parameter_constraints: dict = {
        **BayesianGaussianMixture._parameter_constraints,
        "init_params": [
            StrOptions(
                {"kmeans", "random", "random_from_data", "k-means++", "kmeans-sklearn"}
            )
        ],
    }

    def __init__(
        self,
        *,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None,
        mean_prior=None,
        degrees_of_freedom_prior=None,
        covariance_prior=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
            mean_precision_prior=mean_precision_prior,
            mean_prior=mean_prior,
            degrees_of_freedom_prior=degrees_of_freedom_prior,
            covariance_prior=covariance_prior,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        self._random_seed_for_dask = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        # X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                converged = False
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    _, log_resp = self._e_step(X)
                    log_resp = log_resp.persist()
                    log_resp_exp = da.exp(log_resp).persist()
                    self._m_step(X, log_resp_exp)
                    lower_bound = self._compute_lower_bound(
                        log_resp, None, log_resp_exp
                    )

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        converged = True
                        break

                self._print_verbose_msg_init_end(lower_bound, converged)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter
                    self.converged_ = converged

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                (
                    "Best performing initialization did not converge. "
                    "Try different init parameters, or increase max_iter, "
                    "tol, or check for degenerate data."
                ),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)

    def _check_parameters(self, X):
        """Check that the parameters are well defined.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        mean, cov = da.compute(X.mean(axis=0), da.cov(X.T))
        self._check_weights_parameters()
        self._check_means_parameters(X, mean)
        self._check_precision_parameters(X)
        self._checkcovariance_prior_parameter(X, cov)

    def _check_means_parameters(self, X, mean):
        """Check the parameters of the Gaussian distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.mean_precision_prior is None:
            self.mean_precision_prior_ = 1.0
        else:
            self.mean_precision_prior_ = self.mean_precision_prior

        if self.mean_prior is None:
            self.mean_prior_ = mean
        else:
            self.mean_prior_ = check_array(
                self.mean_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(self.mean_prior_, (n_features,), "means")

    def _checkcovariance_prior_parameter(self, X, cov):
        """Check the `covariance_prior_`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.covariance_type != "full":
            raise NotImplementedError("Only full covariance type is supported")

        if self.covariance_prior is None:
            self.covariance_prior_ = {
                "full": np.atleast_2d(cov),
                # "tied": np.atleast_2d(da.cov(X.T).compute()),
                # "diag": da.var(X, axis=0, ddof=1).compute(),
                # "spherical": da.var(X, axis=0, ddof=1).mean().compute(),
            }[self.covariance_type]

        elif self.covariance_type in ["full", "tied"]:
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(
                self.covariance_prior_,
                (n_features, n_features),
                "%s covariance_prior" % self.covariance_type,
            )
            _check_precision_matrix(self.covariance_prior_, self.covariance_type)
        elif self.covariance_type == "diag":
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(
                self.covariance_prior_,
                (n_features,),
                "%s covariance_prior" % self.covariance_type,
            )
            _check_precision_positivity(self.covariance_prior_, self.covariance_type)
        # spherical case
        else:
            self.covariance_prior_ = self.covariance_prior

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape

        if self.init_params == "kmeans-sklearn":
            # only for verifying correctness on small datasets
            from sklearn import cluster

            X = X.compute()
            resp = np.zeros((n_samples, self.n_components))
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1

            resp = da.from_array(resp)
            X = da.from_array(X)

        elif self.init_params == "kmeans":
            from dask_ml import cluster

            random_state = (
                self._random_seed_for_dask
                if self._random_seed_for_dask is not None
                and isinstance(self._random_seed_for_dask, numbers.Integral)
                else None
            )
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            ).rechunk(chunks=(X.chunks[0],))
            resp = label.map_blocks(
                lambda x: np.eye(self.n_components)[x],
                dtype=np.float64,
                new_axis=1,
                chunks=(label.chunks[0], self.n_components),
            )
        elif self.init_params == "random":
            raise NotImplementedError("Random initialization not implemented")
        elif self.init_params == "random_from_data":
            raise NotImplementedError("Random from data initialization not implemented")
        elif self.init_params == "k-means++":
            raise NotImplementedError("k-means++ initialization not implemented")

        self._initialize(X, resp)

    def _initialize(self, X, resp):
        """Initialization of the mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        nk, xk, sk = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )

        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        _, log_resp = self._estimate_log_prob_resp(X)
        return None, log_resp

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = weighted_log_prob.map_blocks(
            lambda x: logsumexp(x, axis=1), dtype=weighted_log_prob.dtype, drop_axis=1
        )
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - da.expand_dims(log_prob_norm, axis=1)
        return None, log_resp

    def _estimate_log_prob(self, X):
        _, n_features = X.shape
        # We remove `n_features * np.log(self.degrees_of_freedom_)` because
        # the precision matrix is normalized
        log_gauss = _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        log_lambda = n_features * np.log(2.0) + np.sum(
            digamma(
                0.5
                * (self.degrees_of_freedom_ - np.arange(0, n_features)[:, np.newaxis])
            ),
            0,
        )

        return log_gauss + 0.5 * (log_lambda - n_features / self.mean_precision_)

    def _m_step(self, X, log_resp_exp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape

        nk, xk, sk = _estimate_gaussian_parameters(
            X, log_resp_exp, self.reg_covar, self.covariance_type
        )
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _compute_lower_bound(self, log_resp, log_prob_norm, log_resp_exp):
        """Estimate the lower bound of the model.

        The lower bound on the likelihood (of the training data with respect to
        the model) is used to detect the convergence and has to increase at
        each iteration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        log_prob_norm : float
            Logarithm of the probability of each sample in X.

        Returns
        -------
        lower_bound : float
        """
        # Contrary to the original formula, we have done some simplification
        # and removed all the constant terms.
        (n_features,) = self.mean_prior_.shape

        # We removed `.5 * n_features * np.log(self.degrees_of_freedom_)`
        # because the precision matrix is normalized.
        log_det_precisions_chol = _compute_log_det_cholesky(
            self.precisions_cholesky_, self.covariance_type, n_features
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        if self.covariance_type == "tied":
            log_wishart = self.n_components * np.float64(
                _log_wishart_norm(
                    self.degrees_of_freedom_, log_det_precisions_chol, n_features
                )
            )
        else:
            log_wishart = np.sum(
                _log_wishart_norm(
                    self.degrees_of_freedom_, log_det_precisions_chol, n_features
                )
            )

        if self.weight_concentration_prior_type == "dirichlet_process":
            log_norm_weight = -np.sum(
                betaln(self.weight_concentration_[0], self.weight_concentration_[1])
            )
        else:
            log_norm_weight = _log_dirichlet_norm(self.weight_concentration_)

        return (
            -da.sum(log_resp_exp * log_resp).compute()
            - log_wishart
            - log_norm_weight
            - 0.5 * n_features * np.sum(np.log(self.mean_precision_))
        )

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        check_is_fitted(self)
        # X = self._validate_data(X, reset=False)
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        """Evaluate the components' density for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each Gaussian component for each sample in X.
        """
        check_is_fitted(self)
        # X = self._validate_data(X, reset=False)
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)
