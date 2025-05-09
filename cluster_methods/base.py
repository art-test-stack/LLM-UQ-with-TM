from sklearn.metrics import silhouette_score

class ClusterBase:
    """
    Base class for clustering methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize the clustering method with optional parameters.

        Args:
            **kwargs: Optional parameters for the clustering method

        """
    
    def __init__(self, **kwargs):
        raise NotImplementedError("This is an abstract class. Please implement the __init__ method in the subclass.")

    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Please implement the __call__ method in the subclass.")

    def fit(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Please implement the fit method in the subclass.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Please implement the predict method in the subclass.")

    def plot(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Please implement the plot method in the subclass.")

    def compute_silhouette_score(self, X, labels):
        """
        Compute the silhouette score for the clustering.

        Args:
            X (array-like): Input data.
            labels (array-like): Cluster labels.

        Returns:
            float: Silhouette score.
        """

        try:
            return silhouette_score(X, labels)
        except ValueError as e:
            print(f"Error computing silhouette score: {e}")
            return -1