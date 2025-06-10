from pyTsetlinMachine.tools import Binarizer as TmBinarizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from typing import Callable, List, Optional


class Binarizer(TmBinarizer):
    def __repr__(self):
        return f"Binarizer(max_bits_per_feature={self.max_bits_per_feature})"

class MaxThresholdBinarizer(TmBinarizer):
    def __repr__(self):
        return f"MaxThresholdBinarizer(max_bits_per_feature={self.max_bits_per_feature})"
    
    def transform(self, X):
        X_transformed = np.zeros((X.shape[0], self.number_of_features))
        pos = 0
        for i in range(X.shape[1]):
            thresholds = self.unique_values[i]
            for row in range(X.shape[0]):
                value = X[row, i]
                max_idx = -1
                for j, threshold in enumerate(thresholds):
                    if value >= threshold:
                        max_idx = j
                if max_idx >= 0:
                    X_transformed[row, pos + max_idx] = 1
            pos += thresholds.size

        return X_transformed

class AugmentedBinarizer:
    def __init__(self, max_bits_per_feature=2):
        self.max_bits_per_feature = max_bits_per_feature
        self.max_thresholds = (1 << max_bits_per_feature) - 1  # 2^bits - 1

    def __repr__(self):
        return f"AugmentedBinarizer(max_bits_per_feature={self.max_bits_per_feature})"

    def fit(self, X):
        self.unique_values = []
        for i in range(X.shape[1]):
            uv = np.unique(X[:, i])[1:]  # Exclude minimum value
            if uv.size > self.max_thresholds:
                step_size = uv.size / self.max_thresholds
                selected_thresholds = []
                pos = 0.0
                while len(selected_thresholds) < self.max_thresholds and int(pos) < uv.size:
                    selected_thresholds.append(uv[int(pos)])
                    pos += step_size
                thresholds = np.array(selected_thresholds)
            else:
                thresholds = uv
            self.unique_values.append(thresholds)
        return self

    def transform(self, X):
        n_samples = X.shape[0]
        n_features = len(self.unique_values)
        n_output_bits = n_features * self.max_bits_per_feature

        X_transformed = np.zeros((n_samples, n_output_bits), dtype=int)

        for i in range(n_features):
            thresholds = self.unique_values[i]
            for sample_idx in range(n_samples):
                # Count thresholds passed
                count = np.sum(X[sample_idx, i] >= thresholds)

                # Encode count in binary
                binary_repr = list(np.binary_repr(count, width=self.max_bits_per_feature))
                binary_values = [int(b) for b in binary_repr]

                start = i * self.max_bits_per_feature
                end = start + self.max_bits_per_feature
                X_transformed[sample_idx, start:end] = binary_values

        return X_transformed

class RollingBinarizer:
    def __init__(self, window_size=10, max_bits_per_feature=25):
        self.window_size = window_size
        self.max_bits_per_feature = max_bits_per_feature
        self.thresholds_per_feature = []

    def fit(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        self.thresholds_per_feature = []

        for feature_idx in range(n_features):
            thresholds = []

            for i in range(n_samples):
                start = max(0, i - self.window_size + 1)
                window = X[start:i+1, feature_idx]
                if window.size == 0:
                    continue

                quantiles = np.linspace(0, 1, self.max_bits_per_feature + 2)[1:-1]
                thresholds.extend(np.quantile(window, quantiles))

            unique_thresholds = np.unique(thresholds)
            self.thresholds_per_feature.append(unique_thresholds)

    def transform(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        total_output_dim = sum(len(t) for t in self.thresholds_per_feature)
        X_transformed = np.zeros((n_samples, total_output_dim))

        pos = 0
        for feature_idx in range(n_features):
            thresholds = self.thresholds_per_feature[feature_idx]
            for threshold in thresholds:
                X_transformed[:, pos] = (X[:, feature_idx] >= threshold).astype(int)
                pos += 1

        return X_transformed


import hashlib

def hash_id(id_val, num_bins):
    """Hash function for a single integer ID into [0, num_bins)."""
    h = hashlib.md5(str(id_val).encode()).hexdigest()
    return int(h, 16) % num_bins

def reduce_id_features(X_ids, num_bins):
    """
    Reduce the multi-hot ID part of X using feature hashing.
    
    Parameters:
    - X_ids: shape (n_samples, n_ids), binary matrix where each row is a multi-hot ID vector
    - num_bins: target dimensionality for reduced ID space

    Returns:
    - X_ids_hashed: shape (n_samples, num_bins), hashed binary matrix
    """
    n_samples, n_ids = X_ids.shape
    X_hashed = np.zeros((n_samples, num_bins), dtype=np.uint8)

    for i in range(n_samples):
        active_ids = np.nonzero(X_ids[i])[0]  # Get indices of 1s (active IDs)
        for idx in active_ids:
            bin_idx = hash_id(idx, num_bins)
            X_hashed[i, bin_idx] = 1  # Set hashed bin to 1
    
    return X_hashed.astype(np.int8)

class DataPreprocessor:
    def __init__(
            self, 
            csv_path: str, 
            binarizer: Optional[Callable] = None,
            columns_to_drop: Optional[List[str]] = None,
            drop_batch_ids: bool = False,
            retrieve_mhe_batch_ids: bool = False, # Should be False for LCTM training but True for LCTM testing
            hash_batch_ids: bool = False,
            verbose: bool = False,
        ):
        self.csv_path = csv_path
        self.binarizer = binarizer
        self.df = pd.read_csv(csv_path).iloc[1:]
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else []
        self._default_columns_to_drop()
        self.verbose = verbose
        self.nb_batch_ids = 0
        self.columns_dropped = []
        self.drop_batch_ids = drop_batch_ids
        self.hash_batch_ids = hash_batch_ids
        self.retrieve_mhe_batch_ids = retrieve_mhe_batch_ids
        # self.binarizer = Binarizer(max_bits_per_feature=max_bits_per_feature)

    def _default_columns_to_drop(self):
        if "epoch" not in self.columns_to_drop:
            self.columns_to_drop.append("epoch")
        if "batch_size" not in self.columns_to_drop:
            self.columns_to_drop.append("batch_size")
        if "batch_ids" not in self.columns_to_drop:
            self.columns_to_drop.append("batch_ids")

    def preprocess(
            self,
            df: pd.DataFrame,
            max_id: Optional[int] = None,
            return_columns: bool = False,
            fit: bool = False,
        ):
        if self.verbose:
            print("df.columns", df.columns)

        # df = df.iloc[1:]
        # df.drop(df[df["epoch"].isin(epochs_to_remove)].index, inplace=True)
        # Drop columns with 'inf' values and print the columns dropped
        # One-hot encode 'batch_ids' column
        is_batch_ids = False

        if (("batch_ids" in df.columns) and (not self.drop_batch_ids) and self.binarizer) or (("batch_ids" in df.columns) and self.retrieve_mhe_batch_ids):
            # Convert string representation of list to actual list
            df["batch_ids"] = df["batch_ids"].apply(lambda x: eval(x) if isinstance(x, str) else [])
            mlb = MultiLabelBinarizer()
            # Multi-hot encode the 'batch_ids' column
            batch_ids_mhe = pd.DataFrame(mlb.fit_transform(df["batch_ids"]), columns=[f"batch_id_{cls}" for cls in mlb.classes_], index=df.index).to_numpy().astype(np.int8)
            self.nb_batch_ids = len(mlb.classes_)
            df.drop(columns=["batch_ids"], inplace=True)
            is_batch_ids = True

        else: 
            self.nb_batch_ids = 0
        for col in self.columns_to_drop:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
                self.columns_dropped.append(col)
            
        cols_with_inf = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
        if cols_with_inf:
            if self.verbose:
                print("Columns with 'inf' values dropped:", cols_with_inf)
            df.drop(columns=cols_with_inf, inplace=True)
            self.columns_dropped += cols_with_inf
        columns = df.columns
        X = df.to_numpy(copy=True)
        if max_id is not None:
            X = X[:max_id,:]
            if is_batch_ids or self.retrieve_mhe_batch_ids:
                batch_ids_mhe = batch_ids_mhe[:max_id]
        if self.verbose:
            print("X.shape", X.shape)

        if self.binarizer:
            # Check if the data is already binarized
            if fit:
                self.binarizer.fit(X)
            X_train = self.binarizer.transform(X)
            X_train = X_train.astype(np.int8)
        else:
            X_train = pd.DataFrame(X, columns=columns)
        
        if is_batch_ids and not self.drop_batch_ids:
            if self.hash_batch_ids:
                num_bins = 256 # Set the number of bins for hashing
                hashed_batch_ids = reduce_id_features(batch_ids_mhe, num_bins)
                X_train = np.concatenate([hashed_batch_ids, X_train], axis=1)
            else:
                X_train = np.concatenate([batch_ids_mhe, X_train], axis=1)

        if self.verbose:
            print("X_train.shape", X_train.shape)
        if return_columns and self.retrieve_mhe_batch_ids:
            return X_train, columns, batch_ids_mhe
        elif return_columns:
            return X_train, columns
        elif self.retrieve_mhe_batch_ids:
            return X_train, batch_ids_mhe
        else:
            return X_train
    
    def fit(
            self,
            df: Optional[pd.DataFrame] = None,
            max_id: Optional[int] = None,
            return_columns: bool = False,
        ):
        # Fit the binarizer to the data
        if df is None:
            df = self.df
        return self.preprocess(df=df, max_id=max_id, return_columns=return_columns, fit=True)

    def transform(
            self,
            df: Optional[pd.DataFrame] = None,
            max_id: Optional[int] = None,
            return_columns: bool = False,
        ):
        # Fit the binarizer to the data
        if df is None:
            df = self.df
        return self.preprocess(df=df, max_id=max_id, return_columns=return_columns, fit=True)
    
    def fit_transform(
            self,
            df: Optional[pd.DataFrame] = None,
            max_id: Optional[int] = None,
            return_columns: bool = False,
        ):
        # Fit the binarizer to the data and transform it
        if df is None:
            df = self.df
        return self.preprocess(df=df, max_id=max_id, return_columns=return_columns, fit=True)