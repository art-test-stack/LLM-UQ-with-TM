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


def pipeline_lctm(csv_path: Union[str,Path] = "dataset/uq_big-model1.csv"):
    num_clauses_l = [8]
    T_l = [5]
    S_l = [100]
    num_features = 600
    num_subpatterns = 10
    min_samples_per_sub_pattern = 700
    max_samples_per_sub_pattern = 700
    reset_guess_threshold = 200 # if LAs assigned to label decision stayed in a loop more than that threshold without being rewarded as all together at single loop. We resample and reinitialize these LAs with new random initial labels and states.
    pattern_search_perc = 1.0 # percentage of all labels learning automata that must be rewarded together to accept their labels
    epsilon =0.8 # percentage of how likely to penalize a wrong labeled decided from an LA. leaving a window for exploration (1 - epsilon). If all LAs penalized at once for example, they will be in a loop due to the symetric nature of the issue.

    df = pd.read_csv(csv_path)
    epochs_to_remove = [0]
    df.drop(df[df["epoch"].isin(epochs_to_remove)].index, inplace=True)
    X = df.to_numpy(copy=True)

    b = Binarizer(max_bits_per_feature = 10)
    b.fit(X)
    X_train = b.transform(X)

    results = []
    sil_scores = []
    runs = 1
    while runs != 0:
        for num_clauses in num_clauses_l:
            for T in T_l:
                if T > num_clauses:
                    continue
                for S in S_l:
                    good_counter = 0
                    lctm = LCTM(num_clauses, T, S, platform='CUDA', pattern_search_exit= pattern_search_perc, epsilon = epsilon, reset_guess_threshold = reset_guess_threshold)

                    lctm.fit(num_clauses, num_features, [X_train])
                    print('\nFinal Results--> (Grouped Samples):')
                    all_samples = []
                    all_labels  = []
                    for k, v in lctm.grouped_samples.items():
                        for sample in v:
                            all_samples.append(sample)
                            all_labels.append(k)
                        print('Cluster Size: ', len(v))
                        #print(v)    
                        print('Cluster Included Patterns Info:')
                        good = lctm.get_cluster_info(all_patterns, v)
                        if good:
                            good_counter += 1
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
                    if good_counter == num_subpatterns:
                        results.append(1)
                        #print('Parameters are good: ', num_clauses, T, S)
                        #exit()
                    else:
                        results.append(0)
                        
                        
        runs -= 1
    
    print('Percentage of Successful Learning was ---->', np.mean(results))
    print('SIL Score:  ---->', np.mean(sil_scores))
    print('SIL ERROR:  ---->', np.std(sil_scores))
    print('Experiment Setup used [num_features, num_samples_per_subpattern] ---->', num_features, min_samples_per_sub_pattern)

            