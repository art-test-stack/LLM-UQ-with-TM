from pyTsetlinMachine.tools import Binarizer
import pandas as pd
import numpy as np

def preprocess_tm_data(csv_path: str, binarize: bool = True, max_bits_per_feature: int = 10, drop_epoch: bool = True) -> np.ndarray:
    print("csv_path", csv_path)
    df = pd.read_csv(csv_path)
    epochs_to_remove = [0]

    df.drop(df[df["epoch"].isin(epochs_to_remove)].index, inplace=True)
    # Drop columns with 'inf' values and print the columns dropped
    if drop_epoch:  
        df.drop(columns=["epoch"], inplace=True)
    cols_with_inf = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
    if cols_with_inf:
        print("Columns with 'inf' values dropped:", cols_with_inf)
        df.drop(columns=cols_with_inf, inplace=True)

    X = df.to_numpy(copy=True)

    print("X.shape", X.shape)

    if binarize:
        # Check if the data is already binarized
        b = Binarizer(max_bits_per_feature = 10)
        b.fit(X)
        X_train = b.transform(X)
        X_train = X_train.astype(np.int8)
    else:
        X_train = X

    print("X_train.shape", X_train.shape)
    return X_train
    
