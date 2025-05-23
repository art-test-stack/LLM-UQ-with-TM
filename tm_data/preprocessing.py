from pyTsetlinMachine.tools import Binarizer as TmBinarizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from typing import Callable, List, Optional


class Binarizer(TmBinarizer):
    def __repr__(self):
        return f"Binarizer(max_bits_per_feature={self.max_bits_per_feature})"

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

      
# def preprocess_tm_data(
#         csv_path: str, 
#         binarize: bool = True, 
#         max_bits_per_feature: int = 10, 
#         max_id: int = None, 
#         return_columns: bool = False,
#         columns_to_drop: List[str] = None
#     ) -> np.ndarray:
#     print("csv_path", csv_path)
#     df = pd.read_csv(csv_path)
#     # df.drop(columns=["batch_size"], inplace=True)
#     # epochs_to_remove = [0]
#     print("df.columns", df.columns)

#     df = df.iloc[1:]
#     # df.drop(df[df["epoch"].isin(epochs_to_remove)].index, inplace=True)
#     # Drop columns with 'inf' values and print the columns dropped
#     # One-hot encode 'batch_ids' column

#     if "batch_ids" in df.columns:
#         # Convert string representation of list to actual list
#         df["batch_ids"] = df["batch_ids"].apply(lambda x: eval(x) if isinstance(x, str) else [])
#         mlb = MultiLabelBinarizer()
#         batch_ids_ohe = pd.DataFrame(mlb.fit_transform(df["batch_ids"]), columns=[f"batch_id_{cls}" for cls in mlb.classes_], index=df.index)
    
#     df.drop(columns=columns_to_drop, inplace=True)
        
#     cols_with_inf = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
#     if cols_with_inf:
#         print("Columns with 'inf' values dropped:", cols_with_inf)
#         df.drop(columns=cols_with_inf, inplace=True)
#     columns = df.columns
#     X = df.to_numpy(copy=True)
#     if max_id is not None:
#         X = X[:max_id,:]
#         batch_ids_ohe = batch_ids_ohe[:max_id]

#     print("X.shape", X.shape)

#     if binarize:
#         # Check if the data is already binarized
#         b = Binarizer(max_bits_per_feature = max_bits_per_feature)
#         b.fit(X)
#         X_train = b.transform(X)
#         X_train = X_train.astype(np.int8)
#     else:
#         X_train = X
    
#     X_train = np.concat([batch_ids_ohe.to_numpy(), X_train], axis=1)

#     print("X_train.shape", X_train.shape)
#     if return_columns:
#         return X_train, columns
#     else:
#         return X_train
    
class DataPreprocessor:
    def __init__(
            self, 
            csv_path: str, 
            binarizer: Optional[Callable] = None,
            columns_to_drop: Optional[List[str]] = None,
            drop_batch_ids: bool = False,
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
        print("df.columns", df.columns)

        # df = df.iloc[1:]
        # df.drop(df[df["epoch"].isin(epochs_to_remove)].index, inplace=True)
        # Drop columns with 'inf' values and print the columns dropped
        # One-hot encode 'batch_ids' column
        is_batch_ids = False

        if ("batch_ids" in df.columns) and (not self.drop_batch_ids) and self.binarizer:
            # Convert string representation of list to actual list
            df["batch_ids"] = df["batch_ids"].apply(lambda x: eval(x) if isinstance(x, str) else [])
            mlb = MultiLabelBinarizer()
            batch_ids_ohe = pd.DataFrame(mlb.fit_transform(df["batch_ids"]), columns=[f"batch_id_{cls}" for cls in mlb.classes_], index=df.index).to_numpy().astype(np.int8)
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
            print("Columns with 'inf' values dropped:", cols_with_inf)
            df.drop(columns=cols_with_inf, inplace=True)
            self.columns_dropped += cols_with_inf
        columns = df.columns
        X = df.to_numpy(copy=True)
        if max_id is not None:
            X = X[:max_id,:]
            if is_batch_ids:
                batch_ids_ohe = batch_ids_ohe[:max_id]

        print("X.shape", X.shape)

        if self.binarizer:
            # Check if the data is already binarized
            if fit:
                self.binarizer.fit(X)
            X_train = self.binarizer.transform(X)
            X_train = X_train.astype(np.int8)
        else:
            X_train = pd.DataFrame(X, columns=columns)
        
        if is_batch_ids:
            X_train = np.concatenate([batch_ids_ohe, X_train], axis=1)

        print("X_train.shape", X_train.shape)
        if return_columns:
            return X_train, columns
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