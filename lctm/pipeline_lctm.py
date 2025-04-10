from pyTsetlinMachine.tools import Binarizer
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


def pipeline_lctm(args: Namespace):
    csv_path = args.csv_path
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

    interpretability_cl_storage = os.getenv("TM_INTER_CL")
    lctm_name = "example_lctm"
    res_path = Path(interpretability_cl_storage) / lctm_name
    if not res_path.exists():
        res_path.mkdir(parents=True, exist_ok=True)
    # res_path = str(res_path)
    print("Result of interpretability clusters at:", str(res_path))

    generate_res_name = lambda run, num_clauses, T, S: f"run_{run}_clauses_{num_clauses}_T_{T}_S_{S}"

    print("csv_path", csv_path)
    df = pd.read_csv(csv_path)
    epochs_to_remove = [0]
    df.drop(df[df["epoch"].isin(epochs_to_remove)].index, inplace=True)
    # Drop columns with 'inf' values and print the columns dropped

    cols_with_inf = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
    if cols_with_inf:
        print("Columns with 'inf' values dropped:", cols_with_inf)
        df.drop(columns=cols_with_inf, inplace=True)

    X = df.to_numpy(copy=True)

    print("X.shape", X.shape)

    b = Binarizer(max_bits_per_feature = 10)
    b.fit(X)
    X_train = b.transform(X)

    print("X_train.shape", X_train.shape)
    X_train = X_train.astype(np.int8)
    num_features = X_train.shape[1]

    results = []
    sil_scores = []
    runs = 10
    while runs != 0:
        for num_clauses in num_clauses_l:
            for T in T_l:
                if T > num_clauses:
                    continue
                for S in S_l:
                    lctm = LCTM(num_clauses, T, S, platform='CUDA', pattern_search_exit= pattern_search_perc, epsilon = epsilon, reset_guess_threshold = reset_guess_threshold)

                    lctm.fit(num_clauses, num_features, [X_train])
                    print('\nFinal Results--> (Grouped Samples):')
                    all_samples = []
                    all_labels  = []
                    
                    res_name = generate_res_name(runs, num_clauses, T, S)
                    res = {
                        'num_clauses': num_clauses,
                        'T': T,
                        'S': S,
                        'interpretability_clauses': lctm.interpretability_clauses,
                        'grouped_samples': lctm.grouped_samples
                    }
                    with open(res_path.joinpath(res_name), 'wb') as f:
                        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
                        print('Interpretability Clauses saved at:', res_path.joinpath(res_name))

                    print('Interpretability Clauses:', lctm.interpretability_clauses)
                    print('Grouped Samples:', lctm.grouped_samples)

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
                            score = silhouette_score(all_samples, all_labels, metric='euclidean')
                        except:
                            score = -1
                    # to extract learning (total penalities over inside loops of the LCTM)        
                    '''for loop_index, info in lctm.learning_info.items():
                        print('Loop: ', loop_index)
                        print(info)
                        print('-----------------------')'''
                    sil_scores.append(score)
                    # if good_counter == num_subpatterns:
                    #     results.append(1)
                    #     #print('Parameters are good: ', num_clauses, T, S)
                    #     #exit()
                    # else:
                    #     results.append(0)
                        
                        
        runs -= 1
    
    print('Percentage of Successful Learning was ---->', np.mean(results))
    print('SIL Score:  ---->', np.mean(sil_scores))
    print('SIL ERROR:  ---->', np.std(sil_scores))
    print('Experiment Setup used [num_features, num_samples_per_subpattern] ---->', num_features, min_samples_per_sub_pattern)

            