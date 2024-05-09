
import warnings
import numpy as np
import scipy.sparse as sp
from .sklearn_import.metrics.pairwise import euclidean_distances
from .sklearn_import.utils.extmath import row_norms, squared_norm, cartesian
from .sklearn_import.utils.validation import check_array, check_random_state, as_float_array, check_is_fitted
from joblib import Parallel
from joblib import delayed
import numpy as np
from scipy.optimize import linear_sum_assignment

# Internal scikit learn methods imported into this project
from k_means_constrained.sklearn_import.cluster._k_means import _centers_dense, _centers_sparse
from k_means_constrained.sklearn_import.cluster.k_means_ import _validate_center_shape, _tolerance, KMeans, \
    _init_centroids


def k_means_constrained(X, n_clusters, size_min=None, size_max=None, init='k-means++',
                        n_init=10, max_iter=300, verbose=False,
                        tol=1e-4, random_state=None, copy_x=True, n_jobs=1,
                        return_n_iter=False):
   
    if sp.issparse(X):
        raise NotImplementedError("Not implemented for sparse X")

    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    # Validate init array
    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # subtract of mean of x for more accurate distance computations
    if not sp.issparse(X):
        X_mean = X.mean(axis=0)
        # The copy was already done above
        X -= X_mean

        if hasattr(init, '__array__'):
            init -= X_mean

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    best_labels, best_inertia, best_centers = None, None, None

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_constrained_single(
                X, n_clusters,
                size_min=size_min, size_max=size_max,
                max_iter=max_iter, init=init, verbose=verbose, tol=tol,
                x_squared_norms=x_squared_norms, random_state=random_state)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(kmeans_constrained_single)(X, n_clusters,
                                               size_min=size_min, size_max=size_max,
                                               max_iter=max_iter, init=init,
                                               verbose=verbose, tol=tol,
                                               x_squared_norms=x_squared_norms,
                                               # Change seed to ensure variety
                                               random_state=seed)
            for seed in seeds)
        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if not sp.issparse(X):
        if not copy_x:
            X += X_mean
        best_centers += X_mean

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


def kmeans_constrained_single(X, n_clusters, demand, size_min=None, size_max=None,
                               init='k-means++', max_iter=300, tol=1e-4,
                               verbose=False, random_state=None, x_squared_norms=None):
    """
    Performs a single iteration of k-means clustering with custom distance and penalty.

    Args:
        X (numpy.ndarray): Data points (latitude, longitude) with shape (n_samples, 2).
        n_clusters (int): The number of clusters to form.
        demand (numpy.ndarray): Demand associated with each zip code (shape: (n_samples,)).
        size_min (int, optional): Minimum size for each cluster (unused in this approach). Defaults to None.
        size_max (int, optional): Maximum size for each cluster (unused in this approach). Defaults to None.
        init (str or numpy.ndarray, optional): Method for initialization. Defaults to 'k-means++'.
        max_iter (int, optional): Maximum number of iterations. Defaults to 300.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-4.
        verbose (bool, optional): Verbosity mode. Defaults to False.
        random_state (int, optional): Seed for random number generation. Defaults to None.
        x_squared_norms (numpy.ndarray, optional): Precomputed squared norms of data points for efficiency (if available). Defaults to None.

    Returns:
        tuple: (labels, inertia, centers, n_iter)
            - labels (numpy.ndarray): Labels for each data point (assigned cluster).
            - inertia (float): Sum of squared distances of samples to their closest centers.
            - centers (numpy.ndarray): Cluster centers.
            - n_iter (int): Number of iterations required to converge.
    """

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    n_samples, n_features = X.shape

    # Validate init shape
    if hasattr(init, '__array__'):
        _validate_center_shape(X, n_clusters, init)

    random_state = check_random_state(random_state)
    centers = init

    # Ensure demand array has the same length as the number of samples
    demand = np.asarray(demand)  # Convert to NumPy array if necessary
    if demand.shape[0] != n_samples:
        raise ValueError("Demand array must have the same length as the number of data points (samples).")

    # Main k-means loop
    labels = np.zeros(n_samples, dtype=np.int32)
    prev_labels = np.empty(n_samples, dtype=np.int32)
    inertia = np.finfo(float).max

    # Define a weight factor for balancing distance and demand in the cost calculation
    weight_factor = 1.0  # You can adjust this weight based on your prioritization

    for n_iter in range(max_iter):
        old_center = centers.copy()

        # E-step: Assign data points to closest clusters (considering both distance and demand)
        euclidean_distances_ = euclidean_distances(X, centers)
        squared_distances = euclidean_distances_**2

        # Calculate weighted distance based on location (Euclidean) and normalized demand difference
        normalized_demand = demand / demand.sum()
        demand_diff = np.abs(normalized_demand[:, None] - normalized_demand[None, :])
        weighted_distances = weight_factor * squared_distances + demand_diff

        # Solve minimum cost assignment problem with linear_sum_assignment for efficient cluster assignment
        rows, cols = linear_sum_assignment(weighted_distances)
        labels[:] = cols

        # Check for convergence (all labels unchanged)
        if np.all(labels == prev_labels):
            break

        prev_labels = labels.copy()

        # M-step: Update cluster centers based on assigned data points
        centers = np.array([X[labels == i].mean(axis=0)
                            for i in range(n_clusters)])


    return best_labels, best_inertia, best_centers, i + 1


class KMeansConstrained(KMeans):
   

    def __init__(self, n_clusters=8, size_min=None, size_max=None, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                 verbose=False, random_state=None, copy_x=True, n_jobs=1):

        self.size_min = size_min
        self.size_max = size_max

        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                         verbose=verbose, random_state=random_state, copy_x=copy_x, n_jobs=n_jobs)

    def fit(self, X, y=None):
      
        if sp.issparse(X):
            raise NotImplementedError("Not implemented for sparse X")

        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means_constrained(
                X, n_clusters=self.n_clusters,
                size_min=self.size_min, size_max=self.size_max,
                init=self.init,
                n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs,
                return_n_iter=True)
        return self


def predict(self, X, size_min='init', size_max='init'):
    """
    Predict the closest cluster each sample in X belongs to given the provided constraints.

    The constraints can be temporally overridden when determining which cluster each datapoint is assigned to.

    Only computes the assignment step. It does not re-fit the cluster positions.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        New data to predict.

    size_min : int, optional, default: size_min provided with initialisation
        Constrain the label assignment so that each cluster has a minimum
        size of size_min. If None, no constrains will be applied.
        If 'init' the value provided during initialisation of the
        class will be used.

    size_max : int, optional, default: size_max provided with initialisation
        Constrain the label assignment so that each cluster has a maximum
        size of size_max. If None, no constrains will be applied.
        If 'init' the value provided during initialisation of the
        class will be used.

    Returns
    -------
    labels : array, shape [n_samples,]
        Index of the cluster each sample belongs to.
    """

    if sp.issparse(X):
        raise NotImplementedError("Not implemented for sparse X")

    if size_min == 'init':
        size_min = self.size_min
    if size_max == 'init':
        size_max = self.size_max

    n_clusters = self.n_clusters
    n_samples = X.shape[0]

    check_is_fitted(self, 'cluster_centers_')

    X = self._check_test_data(X)

    # Allocate memory to store the distances (optional for efficiency)
    # distances = np.zeros(shape=(n_samples,), dtype=X.dtype)

    # Call kmeans_constrained_single for label assignment with constraints
    labels, _, _ = kmeans_constrained_single(X, n_clusters, demand=None, size_min=size_min, size_max=size_max,
                                              init=self.cluster_centers_)  # Assuming no demand for predict

    return labels


    def fit_predict(self, X, y=None):

        return self.fit(X).labels_
