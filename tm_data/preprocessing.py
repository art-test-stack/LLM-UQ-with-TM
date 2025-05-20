from pyTsetlinMachine.tools import Binarizer
import pandas as pd
import numpy as np

def preprocess_tm_data(
        csv_path: str, 
        binarize: bool = True, 
        max_bits_per_feature: int = 10, 
        drop_epoch: bool = True, 
        max_id: int = None, 
        return_columns: bool = False
    ) -> np.ndarray:
    print("csv_path", csv_path)
    df = pd.read_csv(csv_path)
    # df.drop(columns=["batch_size"], inplace=True)
    # epochs_to_remove = [0]
    print("df.columns", df.columns)

    df = df.iloc[1:]
    # df.drop(df[df["epoch"].isin(epochs_to_remove)].index, inplace=True)
    # Drop columns with 'inf' values and print the columns dropped
    if drop_epoch:  
        # df.drop(columns=["epoch","grad_cos_dist","batch_ids"], inplace=True)
        df.drop(columns=["epoch","batch_ids","grad_median"], inplace=True)
    print("df.columns", df.columns)
    cols_with_inf = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
    if cols_with_inf:
        print("Columns with 'inf' values dropped:", cols_with_inf)
        df.drop(columns=cols_with_inf, inplace=True)
    columns = df.columns
    X = df.to_numpy(copy=True)
    if max_id is not None:
        X = X[:max_id,:]

    print("X.shape", X.shape)

    if binarize:
        # Check if the data is already binarized
        b = Binarizer(max_bits_per_feature = max_bits_per_feature)
        b.fit(X)
        X_train = b.transform(X)
        X_train = X_train.astype(np.int8)
    else:
        X_train = X

    print("X_train.shape", X_train.shape)
    if return_columns:
        return X_train, columns
    else:
        return X_train
    
