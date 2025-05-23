from llm.utils import get_model_dir
from tm_data.preprocessing import DataPreprocessor, Binarizer, MaxThresholdBinarizer, AugmentedBinarizer
# from pyTsetlinMachine.tools import Binarizer
from tmu.tsetlin_machine import LCTM

import numpy as np
from time import time
import random
import pandas as pd
from sklearn.metrics import silhouette_score
# from my_datasets.data import Synthetic
from pathlib import Path
from typing import Union
from argparse import Namespace

import pickle
import os
from pathlib import Path

def create_dir(path: Union[str, Path], k: int = 0):
    lctm_name = "example" if k==0 else f"example_{k}"
    res_path = Path(path) / lctm_name
    if res_path.exists():
        return create_dir(path, k + 1)
    else:
        res_path.mkdir(parents=True, exist_ok=True)
        return res_path

def save_results(res_path: Path, res_name: str, res: dict, text_printed: str = "Interpretability Clauses"):
    with open(res_path.joinpath(res_name), 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'{text_printed} saved at:', res_path.joinpath(res_name))
    return res

def pipeline_lctm(args: Namespace):
    # Load the storage directories
    llm_name = args.model
    binarizer = args.binarizer

    llm_dir = get_model_dir(model_name=llm_name)
    if not llm_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist. Please run the training first. Tried to find it at: {llm_dir}")
    
    csv_path = llm_dir.joinpath("fetched_batch_data.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist. Please run the training first. Tryied to find it at: {csv_path}")
    
    interpretability_cl_storage = llm_dir.joinpath("interpretability_clauses")
    if not interpretability_cl_storage.exists():
        interpretability_cl_storage.mkdir(parents=True, exist_ok=True)

    res_path = create_dir(interpretability_cl_storage)
    generate_res_name = lambda run, num_clauses, T, S: f"run_{run}_clauses_{num_clauses}_T_{T}_S_{S}.pkl"

    # Create the grid for training
    num_clauses_l = [8]
    T_l = [5]
    S_l = [100]
    # num_features = 600
    # num_subpatterns = 10
    min_samples_per_sub_pattern = 700
    max_samples_per_sub_pattern = 700
    reset_guess_threshold = 200 # if LAs assigned to label decision stayed in a loop more than that threshold without being rewarded as all together at single loop. We resample and reinitialize these LAs with new random initial labels and states.
    pattern_search_perc = 1.0 # percentage of all labels learning automata that must be rewarded together to accept their labels
    epsilon =0.8 # percentage of how likely to penalize a wrong labeled decided from an LA. leaving a window for exploration (1 - epsilon). If all LAs penalized at once for example, they will be in a loop due to the symetric nature of the issue.

    
    # lctm_name = "example"
    # res_path = Path(interpretability_cl_storage) / lctm_name
    # if res_path.exists():
    #     res_path.mkdir(parents=True, exist_ok=True)

    print("Result of interpretability clusters at:", str(res_path))


    # Load the data
    max_bits_per_feature = 10
    BinarizerCls = {
        "default": Binarizer,
        "max": MaxThresholdBinarizer,
        "augmented": AugmentedBinarizer,
    }[binarizer]
    binarizer = BinarizerCls(max_bits_per_feature=max_bits_per_feature)

    data_prep = DataPreprocessor(
        csv_path=csv_path,
        binarizer=binarizer,
        columns_to_drop=[],
        verbose=True,
    )
    X_train = data_prep.fit_transform()

    num_samples, num_features = X_train.shape

    permutation = np.random.permutation(X_train.shape[1])
    X_train = X_train[:, permutation]
    # Store the hyperparameters
    hyperparameters = {
        'num_clauses': num_clauses_l,
        'T': T_l,
        'S': S_l,
        'min_samples_per_sub_pattern': min_samples_per_sub_pattern,
        'max_samples_per_sub_pattern': max_samples_per_sub_pattern,
        'pattern_search_perc': pattern_search_perc,
        'reset_guess_threshold': reset_guess_threshold,
        'epsilon': epsilon,
        'csv_path': csv_path,
        'num_samples': num_samples,
        'num_features': num_features,
        'binarizer': str(binarizer),
        'max_bits_per_feature': max_bits_per_feature,
        'nb_batch_ids': data_prep.nb_batch_ids,
        'columns_dropped': data_prep.columns_dropped,
        'drop_batch_ids': data_prep.drop_batch_ids,
        'hash_batch_ids': data_prep.hash_batch_ids,
        'retrieve_mhe_batch_ids': data_prep.retrieve_mhe_batch_ids,
    }
    save_results(res_path, 'hyperparameters.pkl', hyperparameters, text_printed='Hyperparameters')
    
    print("Number of samples for training:", num_samples)

    sil_scores = []
    runs = 10
    run = 0
    while run < runs:
        for num_clauses in num_clauses_l:
            for T in T_l:
                if T > num_clauses:
                    continue
                for S in S_l:
                    res_name = generate_res_name(run, num_clauses, T, S)
                    sil_score = train_lctm(
                        X_train,
                        num_clauses,
                        T,
                        S,
                        num_features,
                        res_path,
                        res_name,
                        pattern_search_perc,
                        reset_guess_threshold,
                        epsilon,
                    )
                    sil_scores.append({res_name: sil_score})
                        
        run += 1
    res = [list(sil_score.values())[0] for sil_score in sil_scores]
    print('Best SIL Score ---->', np.max(res))
    print('SIL Score:  ---->', np.mean(res))
    print('SIL ERROR:  ---->', np.std(res))
    print('Experiment Setup used [num_features, num_samples_per_subpattern] ---->', num_features, min_samples_per_sub_pattern)

            
def train_lctm(
        X_train: np.ndarray,
        num_clauses: int,
        T: int,
        S: int,
        num_features: int,
        res_path: Path,
        res_name: str,
        pattern_search_perc: float,
        reset_guess_threshold: int,
        epsilon: float,
    ) -> None:
    lctm = LCTM(
        num_clauses, 
        T, 
        S, 
        platform='CUDA', 
        pattern_search_exit=pattern_search_perc, 
        epsilon = epsilon, 
        reset_guess_threshold=reset_guess_threshold
    )
    lctm.fit(num_clauses, num_features, [X_train])
    
    all_samples = []
    all_labels  = []
    

    for k, v in lctm.grouped_samples.items():
        for sample in v:
            all_samples.append(sample)
            all_labels.append(k)
        print('Cluster Size: ', len(v))
        #print(v)    
        print('Cluster Included Patterns Info:')
        # good = lctm.get_cluster_info(all_patterns, v)
        # if good:
            # good_counter += 1
        print('----------------------------------------')
    try:
        score = float(silhouette_score(all_samples, all_labels, metric='euclidean'))
    except:
        score = -1
    # to extract learning (total penalities over inside loops of the LCTM)        
    '''for loop_index, info in lctm.learning_info.items():
        print('Loop: ', loop_index)
        print(info)
        print('-----------------------')'''
    
    res = {
        'num_clauses': num_clauses,
        'T': T,
        'S': S,
        'interpretability_clauses': lctm.interpretability_clauses,
        'grouped_samples': lctm.grouped_samples,
        'silhouette_score': score,
    }
    save_results(res_path, res_name, res)

    print('Interpretability Clauses:', lctm.interpretability_clauses)
    print('Grouped Samples:', lctm.grouped_samples)

    return score

