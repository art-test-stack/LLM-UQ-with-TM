from tm_data.preprocessing import AugmentedBinarizer, Binarizer, DataPreprocessor
from llm.utils import get_model_dir
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Union, Optional
from pathlib import Path
from enum import Enum
import re


class DataPreprocessed(Enum):
    """
    Enum class to represent the data preprocessed.
    """
    BATCH = "batch"
    EPOCH = "epoch"

class LCTMResults:
    """
    Class to handle the results of the LCTM model.
    """

    def __init__(self, llm_name: str, data: Optional[str] = "batch", verbose: bool = False):
        assert data in DataPreprocessed._value2member_map_, "Data preprocessed not supported. Choose between 'batch' and 'epoch'"

        self.llm_name = llm_name
        self.model_dir = get_model_dir(model_name=llm_name)
        self.interpretability_clauses_dir = self.model_dir.joinpath("interpretability_clauses")
        self.num_folders = sum(1 for p in self.interpretability_clauses_dir.iterdir() if p.is_dir())
        print(f"Number of folders in interpretability_clauses: {self.num_folders}")
        self.init_current_example()
        self.get_path = lambda run: f"{self.current_example}/{run}"

        data = DataPreprocessed(data)
        if data == DataPreprocessed.BATCH:
            self.data_folder = self.model_dir.joinpath("fetched_batch_data.csv")
        elif data == DataPreprocessed.EPOCH:
            self.data_folder = self.model_dir.joinpath("fetched_training_data.csv")
        else:
            raise ValueError("Data preprocessed not supported. Choose between 'batch' and 'epoch'")
        
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder does not exist at: {self.data_folder}.")
        
        self.verbose = verbose
        
        
    def __iter__(self):
        """
        Initialize the iterator.
        """
        self.init_current_example()
        return self
    
    def __next__(self):
        """
        Get the next example.
        """
        return self.next_example()
     
    def init_current_example(self):
        """
        Initialize the current example to 0.
        """
        self.current_example = 0
        self.current_folder = self.interpretability_clauses_dir.joinpath("example")
        if not self.current_folder.exists():
            raise FileNotFoundError(f"Interpretability clauses directory does not exist at: {self.current_folder}.")

    def next_example(self):
        """
        Get the next example from the interpretability clauses directory.
        """
        if self.current_example + 1 > self.num_folders:
            raise StopIteration("No more examples to process.")
        
        self.current_example += 1
        
        self.current_folder = self.interpretability_clauses_dir.joinpath(f"example_{self.current_example}")
        self.current_results = LCTMCurrentResults(self.current_folder)

        return self.current_results
    
    def get_examples(self):
        return self.current_results
    

class LCTMCurrentResults:
    def __init__(self, folder: Union[str, Path], data_folder: Path, verbose: bool = False):
        self.runs = [ f for f in folder.iterdir() if f.is_file() and "run_" in f.name ]
        self.run = self.runs[0]
        self.current = 0
        self.max_runs = len(self.runs)
        self.data_folder = data_folder

        with open(data_folder.joinpath("hyperparameters.pkl"), 'rb') as file:
            self.hyperparameters = pickle.load(file)

        self._get_binarizer()  # Parse the binarizer representation from hyperparameters

        data_preprocesor = DataPreprocessor(
            csv_path=data_folder,
            binarizer=self.binarizer,
            columns_to_drop=[],
            verbose=True,
        )
        self.data, self.columns = data_preprocesor.fit_transform(max_id=self.hyperparameters["num_samples"], return_columns=True)
        self.batch_ids = data_preprocesor.nb_batch_ids
        self.verbose = verbose
        # Parse the binarizer representation from hyperparameters
        
    def _get_binarizer(self):
        binarizer_repr = self.hyperparameters.get('binarizer', '')
        if binarizer_repr.startswith('Binarizer('):
            match = re.search(r'max_bits_per_feature=(\d+)', binarizer_repr)
            max_bits = int(match.group(1)) if match else 1
            self.binarizer = Binarizer(max_bits_per_feature=max_bits)

        elif binarizer_repr.startswith('AugmentedBinarizer('):
            match = re.search(r'max_bits_per_feature=(\d+)', binarizer_repr)
            max_bits = int(match.group(1)) if match else 2
            self.binarizer = AugmentedBinarizer(max_bits_per_feature=max_bits)
        else:
            raise ValueError(f"Unknown binarizer type: {binarizer_repr}")
    
    def __repr__(self):
        return f"LCTMSubResults('{self.run}')"
    
    def __iter__(self):
        return LCTMResult(path=self.run, data=self.data, hyperparameters=self.hyperparameters, verbose=self.verbose)
    
    def __next__(self):
        if self.current + 1 > self.max_runs:
            raise StopIteration("No more runs to process.")
        self.current += 1
        self.run = self.runs[self.current]
        # self.run = self.runs[self.current - 1]
        self.current_results = LCTMResult(self.run, self.data, self.hyperparameters, verbose=self.verbose)
        return self.current_results
 

    # def plot_features_threshold(X, ids_by_class, run, with_features_colors=False, with_sample_colors=False):
    #     for c_id, idx in ids_by_class.items():
    #         plt.figure(figsize=(15, 9))
    #         x = np.arange(X.shape[1]).tolist() 
    #         labels = list(df_columns)

    #         plt.axes().set_xticks(x, labels, rotation=45)
    #         if with_features_colors:
    #             for i in range(X.shape[1]):
    #                 plt.scatter(np.full_like(idx, fill_value=i), X[idx, i])
    #         elif with_sample_colors:
    #             cmap = plt.cm.get_cmap('viridis', len(idx))
    #             for i, sample_idx in enumerate(idx):
    #                 plt.scatter(x, X[sample_idx], color=cmap(i), label=f"Epoch {sample_idx+ 1}")
    #             sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(idx)-1))
    #             sm.set_array([])
    #             # cbar = plt.colorbar(sm, ticks=range(len(idx)))
    #             # cbar.set_label('Sample Index')
    #         else:
    #             plt.scatter(x * X[idx].shape[0], X[idx])
    #         plt.xlabel("Features")
    #         plt.ylabel("Thresholds")
    #         plt.title(f"Scatter plot of class {c_id} on run {run}")
    #         plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    #         plt.show()

    # def plot_current_examples(self):
    #     current_example_runs = [ f for f in self.current_folder.iterdir() if f.is_file()]
    #     for run in current_example_runs:
    #         print("-"*20)
    #         print(f"\nRUN {run}")
    #         print("-"*10)



    #         print("\nSample to Clauses Mapping:")
    #         # for key, clause_lists in sorted_obj.items():
    #         #     sample_clauses_mapping[key] = []
    #         #     for clause_list in clause_lists:
    #         #         associated_samples = set(range(X_train.shape[0]))
    #         #         for c_id, clause in enumerate(clause_list):
    #         #             feature = int(clause.split('x')[1])
    #         #             if clause.startswith('¬'):
    #         #                 associated_samples &= set(np.where(X_train[:, feature] == 0)[0])
    #         #             else:
    #         #                 associated_samples &= set(np.where(X_train[:, feature] == 1)[0])
                    
    #         #             if associated_samples == set():
    #         #                 print("No associated samples")
    #         #                 print("Break at clause:", clause, "with id:", c_id)
    #         #                 break
    #         #         sample_clauses_mapping[key].append(list(associated_samples))
    #         #         unassociated_samples -= associated_samples

    #         # unassociated_samples = list(unassociated_samples)

    #         # print("Sample to Clauses Mapping:", sample_clauses_mapping)
    #         ## Match samples to grouped samples
            

    #         plot_features_threshold(X_t, ids_by_class, run, with_sample_colors=True)

class LCTMResult:
    def __init__(self, path, data: np.ndarray, hyperparameters: dict = None, verbose: bool = False):
        self.verbose = verbose
        self.path = path
        self.hyperparameters = hyperparameters
        with open(path, 'rb') as file:
            obj = pickle.load(file)

        if len(list(obj["interpretability_clauses"].keys())) > 8:
            classes = range(len(obj["grouped_samples"]))
        else:
            classes = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]

        new_obj = { "interpretability_clauses": {}, "grouped_samples": {} }

        for c_id, (k, v) in enumerate(obj["interpretability_clauses"].items()):
            new_obj["interpretability_clauses"][classes[c_id]] = v
            new_obj["grouped_samples"][classes[c_id]] = obj["grouped_samples"][k]

        if self.verbose:
            print("\nNumber of classes:", len(obj["grouped_samples"]))

        new_obj["interpretability_clauses"] = self._delete_empty_strings_in_clauses(new_obj["interpretability_clauses"])
        new_obj["interpretability_clauses"] = self._sort_clauses_by_feature(new_obj["interpretability_clauses"])

        self._get_missing_clauses(new_obj["interpretability_clauses"])

        for k, v in new_obj.items():
            setattr(self, k, v)

        # self.unassociated_samples = self.get_unassociated_samples(data)
        self.ids_by_class, self.unassociated_samples = self.get_ids_by_class(data)

    
    def _delete_empty_strings_in_clauses(self, interpretability_clauses):
        cleaned_str = 0
        cleaned_clauses = {}
        deleted_str = []
        nb_sub_lists = 0
        for k, v in interpretability_clauses.items():
            cleaned_clauses[k] = []
            for v_id, sub in enumerate(v):
                cleaned_sub = []
                for s_id, subsub in enumerate(sub):
                    if subsub:
                        cleaned_sub.append(subsub)
                    else: 
                        deleted_str.append((k, v_id, s_id))
                        cleaned_str += 1
                    nb_sub_lists += 1
                cleaned_clauses[k].append(cleaned_sub)
        if self.verbose:
            print(f"Number of empty strings: {cleaned_str} over {nb_sub_lists} sublists")
            print(f"Empty strings: {deleted_str}")
        return cleaned_clauses
    
    def _sort_clauses_by_feature(self, interpretability_clauses):
        sorted_clauses = {}
        for k, v in interpretability_clauses.items():
            sorted_clauses[k] = []
            for sub in v:
                sorted_clauses[k].append(sorted(sub, key=lambda x: int(x.split('x')[1])))
        return sorted_clauses

    def _get_missing_clauses(self, interpretability_clauses):
        missing_clauses = {}
        for k, v in interpretability_clauses.items():
            missing_clauses[k] = []
            for sub in v:
                clauses = set([ int(clause.split('x')[1]) for clause in sub ])
                miss_cl = list(set(range(220)) - clauses)
                missing_clauses[k].append(sorted(miss_cl))

        self.missing_clauses = missing_clauses
        return missing_clauses
    

    def TODOget_unassociated_samples(self, data: np.ndarray):
        # TODO
        """
        Get the unassociated samples from the interpretability clauses.
        """
        sample_clauses_mapping = {}
        unassociated_samples = set(range(data.shape[0]))

        print("\nSample to Clauses Mapping:")
        for key, clause_lists in self.interpretability_clauses.items():
            sample_clauses_mapping[key] = []
            for clause_list in clause_lists:
                associated_samples = set(range(data.shape[0]))
                for c_id, clause in enumerate(clause_list):
                    feature = int(clause.split('x')[1])
                    if clause.startswith('¬'):
                        associated_samples &= set(np.where(data[:, feature] == 0)[0])
                    else:
                        associated_samples &= set(np.where(data[:, feature] == 1)[0])
                
                    if associated_samples == set():
                        print("No associated samples")
                        print("Break at clause:", clause, "with id:", c_id)
                        break
                sample_clauses_mapping[key].append(list(associated_samples))
                unassociated_samples -= associated_samples

        unassociated_samples = list(unassociated_samples)
        return unassociated_samples
    
    def plot_interpretability_clauses_image(self):
        pass

    def plot_features_threshold(self, data, with_features_colors=False):
        for c_id, idx in self.ids_by_class.items():
            plt.figure(figsize=(15, 9))
            x = np.arange(data.shape[1]).tolist() 
            labels = list(self.columns)

            plt.axes().set_xticks(x, labels, rotation=45)
            if with_features_colors:
                for i in range(data.shape[1]):
                    plt.scatter(np.full_like(idx, fill_value=i), X[idx,i])
            else:
                plt.scatter(x * data[idx].shape[0], data[idx])
            print(f"Scatter plot of class {c_id} on run {self.run}")
            plt.xlabel("Features")
            plt.ylabel("Thresholds")
            plt.show()

    def get_ids_by_class(self, data):
        grouped_samples = {k: v.tolist() for k, v in self.grouped_samples.items()}
        items_by_class = { k: 0 for k in grouped_samples.keys() }
        not_associated = 0
        ids_by_class = { k: [] for k in grouped_samples.keys() }

        X = data.tolist()
        for x in X:
            associated = False
            for k, v in grouped_samples.items():
                if x in v:
                    items_by_class[k] += 1
                    ids_by_class[k].append(X.pop(X.index(x)))
                    # ids_by_class[k].append(X.index(x))
                    associated = True
                    break
            if not associated:
                not_associated += 1

            if self.verbose:
                print("\nItems by class:", items_by_class)
                print("Number of not associated samples:", not_associated)
                print("Not associated samples ids:", X)
                print("ids by class:", ids_by_class)
        return ids_by_class, X
    
    def get_duplicates(self, data):
        duplicates = np.unique([tuple(row) for row in data], axis=0)

        if self.verbose:
            if len(duplicates) < data.shape[0]:
                print(f"Number of duplicate rows: {data.shape[0] - len(duplicates)}")
            else:
                print("No duplicate rows found.")
        for k, v in self.grouped_samples.items():
            duplicates = np.unique([tuple(row) for row in v], axis=0)
            if self.verbose:
                print(f"'{k}': {duplicates.shape}, {len(v)}. There are {len(v) - duplicates.shape[0]} duplicates.")
            if duplicates.shape[0] < len(v):
                row_ids = []
                for duplicate in duplicates.tolist():
                    if duplicate in data.tolist():
                        row_ids.append(data.tolist().index(duplicate))
                if self.verbose:
                    print(f"Row ID of the duplicate in X_train: {row_ids}")
        return duplicates
                    

    def get_max_threshold_by_feature(self, data, max_bytes_by_features: int = 10):
        X_t = []
        if isinstance(data, list):
            data = np.array(data)
        assert data.shape[1, ] % max_bytes_by_features == 0, "the number of features binarized should be a multiple of max_bytes_by_features value"
        nb_features = data.shape[1] // max_bytes_by_features
        for feature in range(nb_features):
            x = data[:,feature * max_bytes_by_features:(feature + 1) * max_bytes_by_features]
            x_t = np.sum(x, axis=1, keepdims=True)
            # x_t = np.argmax(1 - x, axis=1, keepdims=True)
            X_t.append(x_t)
        X_t = np.concat(X_t, axis=1)
        return X_t