from cluster_methods.base import ClusterBase

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

from sklearn import mixture

# from tm_data.preprocessing import preprocess_tm_data
# csv_path = "/Users/arthurtestard/LLM-UQ-with-TM/models/fetched_training_data.csv"

# X = preprocess_tm_data(csv_path, binarize=False, drop_epoch=True) # [:, :10]

# from sklearn.preprocessing import StandardScaler
# X = StandardScaler().fit_transform(X)

# print(X[0])

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    # plt.xlim(-9.0, 5.0)
    # plt.ylim(-3.0, 6.0)
    # plt.xticks(())
    # plt.yticks(())
    plt.title(title)


# Number of samples per component
n_samples = 500

# Generate random sample, two components
# np.random.seed(0)
# C = np.array([[0.0, -0.1], [1.7, 0.4]])
# X = np.r_[
#     np.dot(np.random.randn(n_samples, 2), C),
#     0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
# ]


# Fit a Gaussian mixture with EM using five components
# gmm = mixture.GaussianMixture(n_components=5, covariance_type="full").fit(X)
# plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")

# Fit a Dirichlet process Gaussian mixture using five components
# dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type="full").fit(X)

# plot_results(
#     X,
#     dpgmm.predict(X),
#     dpgmm.means_,
#     dpgmm.covariances_,
#     1,
#     "Bayesian Gaussian Mixture with a Dirichlet process prior",
# )

# plt.show()

class GaussianMixtureCluster(ClusterBase):
    def __init__(self, n_components=5, covariance_type="full", columns=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.gmm = mixture.GaussianMixture(
            n_components=self.n_components, covariance_type=self.covariance_type
        )
        self.trained = False
        self.columns = columns

    def fit(self, X, verbose=True):
        if self.trained:
            raise ValueError("GaussianMixture has already been trained. Please create a new instance to fit again.")
        self.gmm.fit(X)

        self.trained = True
        self.labels = self.gmm.predict(X)

        self.means = self.gmm.means_
        self.covariances = self.gmm.covariances_
        # if verbose:
        #     print(f"GaussianMixtureCluster means: {self.means}")
        #     print(f"GaussianMixtureCluster covariances: {self.covariances}")

        sil = self.compute_silhouette_score(X, self.labels)
        return self.labels, self.means, self.covariances, sil
    
    def predict(self, X):
        if not self.trained:
            raise ValueError("GaussianMixture has not been trained yet. Please fit the model first.")
        return self.gmm.predict(X)
    
    def plot(self, X):
        if not self.trained:
            raise ValueError("GaussianMixture has not been trained yet. Please fit the model first.")
        plot_results(X, self.labels, self.means, self.covariances, 0, "Gaussian Mixture")
        plt.show()
        return self.labels, self.means, self.covariances
    

class BayesianGaussianMixtureCluster(ClusterBase):
    def __init__(self, n_components=5, covariance_type="full", columns=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.dpgmm = mixture.BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            weight_concentration_prior=1e-2,
            weight_concentration_prior_type="dirichlet_process",
            mean_precision_prior=1e-2,
            covariance_prior=1e0 * np.eye(X.shape[1]),
            init_params="kmeans",
            max_iter=100,
            random_state=2,
        )
        self.trained = False
        self.columns = columns

    def fit(self, X, verbose=True):
        if self.trained:
            raise ValueError("BayesianGaussianMixture has already been trained. Please create a new instance to fit again.")
        
        self.dpgmm.fit(X)
        self.trained = True

        self.labels = self.dpgmm.predict(X)
        self.means = self.dpgmm.means_
        self.covariances = self.dpgmm.covariances_
        if verbose:
            print(f"BayesianGaussianMixtureCluster means: {self.means}")
            print(f"BayesianGaussianMixtureCluster covariances: {self.covariances}")

        sil = self.compute_silhouette_score(X, self.labels)
        return self.labels, self.means, self.covariances
    
    def predict(self, X):
        if not self.trained:
            raise ValueError("BayesianGaussianMixture has not been trained yet. Please fit the model first.")
        return self.dpgmm.predict(X)
    
    def plot(self, X):
        if not self.trained:
            raise ValueError("BayesianGaussianMixture has not been trained yet. Please fit the model first.")
        plot_results(X, self.labels, self.means, self.covariances, 1, "Bayesian Gaussian Mixture with a Dirichlet process prior")
        plt.show()
        return self.labels, self.means, self.covariances
    
