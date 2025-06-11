from tm_data.preprocessing import AugmentedBinarizer, Binarizer, MaxThresholdBinarizer, DataPreprocessor
from llm.utils import get_model_dir
from llm.data.dataset import clean_answer, get_answer_formats
import datasets
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Union, Optional, Dict
from matplotlib.colors import ListedColormap, BoundaryNorm
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

    def __init__(self, llm_name: str, verbose: bool = False, plot_titles: bool = False):
        # assert data in DataPreprocessed._value2member_map_, "Data preprocessed not supported. Choose between 'batch' and 'epoch'"

        self.llm_name = llm_name
        self.model_dir = get_model_dir(model_name=llm_name, return_error=True)
        self.interpretability_clauses_dir = self.model_dir.joinpath("interpretability_clauses")
        self.num_folders = sum(1 for p in self.interpretability_clauses_dir.iterdir() if p.is_dir())
        print(f"Number of folders in interpretability_clauses: {self.num_folders}")
        self.get_path = lambda run: f"{self.current_example}/{run}"
        self.plot_titles = plot_titles

        # data = DataPreprocessed(data)
        # if data == DataPreprocessed.BATCH:
        #     self.data_folder = self.model_dir.joinpath("fetched_batch_data.csv")
        # elif data == DataPreprocessed.EPOCH:
        #     self.data_folder = self.model_dir.joinpath("fetched_training_data.csv")
        # else:
        #     raise ValueError("Data preprocessed not supported. Choose between 'batch' and 'epoch'")
        
        # if not self.data_folder.exists():
        #     raise FileNotFoundError(f"Data folder does not exist at: {self.data_folder}.")
        
        self.verbose = verbose
        self.init_current_example()

    def __iter__(self):
        """
        Initialize the iterator.
        """
        self.init_current_example()
        return self.current_res
    
    def __next__(self):
        """
        Get the next example.
        """
        return self.next_example()
    
    def has_next(self):
        """
        Check if there is a next example.
        """
        return self.current_example + 1 <= self.num_folders
    
    def __len__(self):
        """
        Get the number of examples.
        """
        return self.num_folders
    
    def __getitem__(self, idx):
        """
        Get the example at index idx.
        """
        if idx < 0 or idx >= self.num_folders:
            raise IndexError("Index out of range.")
        self.current_example = idx
        if self.current_example == 0:
            self.current_folder = self.interpretability_clauses_dir.joinpath("example")
        else:
            self.current_folder = self.interpretability_clauses_dir.joinpath(f"example_{self.current_example}")
        if not self.current_folder.exists():
            raise FileNotFoundError(f"Interpretability clauses directory does not exist at: {self.current_folder}.")
        if self.verbose:
            print(f"Current folder: {self.current_folder}")
        return self.get_current_example()
    
    def get_current_example(self):
        self.current_res = LCTMCurrentResults(
            folder=self.current_folder, 
            model_dir=self.model_dir, 
            verbose=self.verbose, 
            plot_titles=self.plot_titles
        )
        return self.current_res

    def init_current_example(self):
        """
        Initialize the current example to 0.
        """
        self.current_example = 0
        self.current_folder = self.interpretability_clauses_dir.joinpath("example")
        if not self.current_folder.exists():
            raise FileNotFoundError(f"Interpretability clauses directory does not exist at: {self.current_folder}.")
        self.get_current_example()
        return self.current_res

    def next_example(self):
        """
        Get the next example from the interpretability clauses directory.
        """
        if self.current_example + 1 > self.num_folders:
            raise StopIteration("No more examples to process.")
        
        self.current_example += 1
        
        self.current_folder = self.interpretability_clauses_dir.joinpath(f"example_{self.current_example}")
        if self.verbose:
            print(f"Current folder: {self.current_folder}")
            # print(f"Data folder: {self.data_folder}")
        self.get_current_example()
        return self.current_res
    
    def get_examples(self):
        return self.current_res
    

class LCTMCurrentResults:
    def __init__(self, folder: Union[str, Path], model_dir: Path, verbose: bool = False, plot_titles: bool = False):
        self.runs = [ f for f in folder.iterdir() if f.is_file() and "run_" in f.name ]

        self.verbose = verbose
        if len(self.runs) > 0:
            self.run = self.runs[0] 
        else:
            print(f"No runs found in {folder}.")
            self.run = None
            return None
        self.current = 0
        self.max_runs = len(self.runs)
        self.model_dir = model_dir
        self.plot_titles = plot_titles
        # print(f"Data folder: {self.data_folder}")
        hp_file = folder.joinpath("hyperparameters.pkl") if folder.joinpath("hyperparameters.pkl").exists() else model_dir.parent.joinpath("default_hyperparameters.pkl")
        
        with open(hp_file, 'rb') as file:
            self.hyperparameters: Dict = pickle.load(file)
        
        document = self.hyperparameters.get("document", "batch")
        self.hyperparameters["drop_batch_ids"] = self.hyperparameters.get("drop_batch_ids", False)

        if self.verbose:
            print(f"Current folder: {folder}")
            print(f"Hyperparameters file: {hp_file}")
            print(f"Hyperparameters: {self.hyperparameters}")
        if document == "batch":
            self.data_folder = self.model_dir.joinpath("fetched_batch_data.csv")
        elif document == "epoch":
            self.data_folder = self.model_dir.joinpath("fetched_training_data.csv")
        else:
            raise ValueError(f"Document {document} not supported. Choose between 'batch' and 'epoch'.")
        
        self._get_binarizer()
        self._prepare_data()  # Parse the binarizer representation from hyperparameters
        self.nb_batch_ids = self.hyperparameters["nb_batch_ids"] if not self.hyperparameters["drop_batch_ids"] else 0
        # Parse the binarizer representation from hyperparameters
    
    def _prepare_data(self):
        data_preprocesor = DataPreprocessor(
            csv_path=self.data_folder,
            binarizer=self.binarizer,
            columns_to_drop=self.hyperparameters["columns_dropped"],
            drop_batch_ids=self.hyperparameters["drop_batch_ids"],
            hash_batch_ids=self.hyperparameters.get("hash_batch_ids", False),
            verbose=self.verbose,
            retrieve_mhe_batch_ids=True
        )
        self.data, self.columns, self.mhe_batch_ids = data_preprocesor.fit_transform(max_id=self.hyperparameters["num_samples"], return_columns=True)
        self.nb_batch_ids = data_preprocesor.nb_batch_ids
        data_preprocesor = DataPreprocessor(
            csv_path=self.data_folder,
            binarizer=None,
            columns_to_drop=self.hyperparameters["columns_dropped"],
            verbose=False,
        )
        data_preprocesor.columns_to_drop.remove("epoch")
        self.raw_data = data_preprocesor.fit_transform(max_id=self.hyperparameters["num_samples"], return_columns=False)

    def _get_binarizer(self):
        binarizer_repr = self.hyperparameters.get('binarizer', '')

        if binarizer_repr.startswith('Binarizer('):
            binarizer_cls = Binarizer
            self.binarizer_repr = "Binarizer"

        elif binarizer_repr.startswith('AugmentedBinarizer('):
            binarizer_cls = AugmentedBinarizer
            self.binarizer_repr = "AugmentedBinarizer"

        elif binarizer_repr.startswith('MaxThresholdBinarizer('):
            binarizer_cls = MaxThresholdBinarizer
            self.binarizer_repr = "MaxThresholdBinarizer"

        else:
            raise ValueError(f"Unknown binarizer type: {binarizer_repr}")
        
        match = re.search(r'max_bits_per_feature=(\d+)', binarizer_repr)
        max_bits = int(match.group(1)) if match else self.hyperparameters.get("max_bits_per_feature", None)
        self.binarizer = binarizer_cls(max_bits_per_feature=max_bits)

        if "max_bits_per_feature" not in self.hyperparameters:
            self.hyperparameters["max_bits_per_feature"] = self.binarizer.max_bits_per_feature
        assert self.hyperparameters["max_bits_per_feature"] == self.binarizer.max_bits_per_feature, f"Hyperparameters and binarizer max_bits_per_feature do not match. Got: {self.hyperparameters['max_bits_per_feature']} and {self.binarizer.max_bits_per_feature}"
    
    def __repr__(self):
        return f"LCTMCurrentResults('{self.run}')"
    
    def __iter__(self):
        self.current = 0
        self._get_current()
        return self
    
    def __next__(self):
        if self.current + 1 >= self.max_runs:
            print("Hello")
            raise StopIteration("No more runs to process.")
        self.current += 1
        self._get_current()
        return self.current_res
    
    def has_next(self):
        """
        Check if there is a next run.
        """
        return self.current + 1 < self.max_runs
    
    def __len__(self):
        """
        Get the number of runs.
        """
        return self.max_runs
    
    def __getitem__(self, idx):
        """
        Get the run at index idx.
        """
        if idx < 0 or idx >= self.max_runs:
            raise IndexError("Index out of range.")
        self.current = idx
        self._get_current()
        return self.current_res
    
    def _get_current(self):
        self.run = self.runs[self.current]
        self.current_res = LCTMResult(path=self.run, data=self.data, hyperparameters=self.hyperparameters, verbose=self.verbose)
        return self.current_res

    def get_max_threshold_by_feature(self):
        X_t = []
        if isinstance(self.data, list):
            data = np.array(self.data)
        else:
            data = self.data
        max_bits_per_feature = self.hyperparameters["max_bits_per_feature"]
        nb_batch_ids = self.nb_batch_ids
        assert (data.shape[1] - nb_batch_ids) % max_bits_per_feature == 0, f"The number of features binarized should be a multiple of max_bits_per_feature value. Got: {data.shape[1] - nb_batch_ids} and {max_bits_per_feature}"
        nb_features = (data.shape[1] - nb_batch_ids) // max_bits_per_feature
        for feature in range(nb_features):
            x = data[:,feature * max_bits_per_feature + nb_batch_ids:(feature + 1) * max_bits_per_feature + nb_batch_ids]
            x_t = np.sum(x, axis=1, keepdims=True)
            # x_t = np.argmax(1 - x, axis=1, keepdims=True)
            X_t.append(x_t)
        X_t = np.concat(X_t, axis=1)
        return X_t 
    
    def get_threshold_index_by_feature(self):
        X_T = []
        if isinstance(self.data, list):
            data = np.array(self.data)
        else:
            data = self.data
        max_bits_per_feature = self.hyperparameters["max_bits_per_feature"]
        nb_batch_ids = self.nb_batch_ids
        assert (data.shape[1] - nb_batch_ids) % max_bits_per_feature == 0, f"The number of features binarized should be a multiple of max_bits_per_feature value. Got: {data.shape[1] - nb_batch_ids} and {max_bits_per_feature}"
        nb_features = (data.shape[1] - nb_batch_ids) // max_bits_per_feature
        for feature in range(nb_features):
            x = data[:,feature * max_bits_per_feature + nb_batch_ids:(feature + 1) * max_bits_per_feature + nb_batch_ids]
            x_t = np.argmax(x, axis=1, keepdims=True)
            X_T.append(x_t)
        X_T = np.concatenate(X_T, axis=1)
        return X_T
 
    def plot_features_threshold(self, with_features_colors=False):
        data = self.get_threshold_index_by_feature() if self.binarizer_repr == "MaxThresholdBinarizer" else self.get_max_threshold_by_feature()

        for c_id, idx in self.current_res.ids_by_class.items():
            plt.figure(figsize=(15, 9))
            x = np.arange(data.shape[1]).tolist() 
            labels = list(self.columns)

            plt.axes().set_xticks(x, labels, rotation=45)
            if with_features_colors:
                for i in range(data.shape[1]):
                    plt.scatter(np.full_like(idx, fill_value=i), data[idx,i])
            else:
                plt.scatter(x * data[idx].shape[0], data[idx])
            print(f"Scatter plot of class {c_id} on run {self.run}")
            plt.xlabel("Features")
            plt.ylabel("Thresholds")
            plt.show()

    def plot_threshold_features(self, with_features_colors=False):
        data = self.get_threshold_index_by_feature() if self.binarizer_repr == "MaxThresholdBinarizer" else self.get_max_threshold_by_feature()

        for c_id, idx in self.current_res.ids_by_class.items():
            plt.figure(figsize=(15, 9))
            labels = list(self.columns)
            y = np.arange(data.shape[1])
            plt.yticks(y, labels, rotation=0)
            if with_features_colors:
                for i in range(data.shape[1]):
                    plt.scatter(data[idx, i], [i] * len(idx), label=labels[i] if i < len(labels) else f"Feature {i}")
            else:
                for i in range(data.shape[1]):
                    plt.scatter(data[idx, i], [i] * len(idx))
            print(f"Scatter plot of thresholds by feature for class {c_id} on run {self.run}")
            plt.ylabel("Features")
            plt.xlabel("Thresholds")
            plt.show()

    def plot_clauses_matrix(self):
        """
        Plot a matrix for each class and clause, where rows are features, columns are bits (max_bits_per_feature),
        and cells are colored green if x_i is present (True), red if ¬x_i (False), and white if not present.
        """
        max_bits_per_feature = self.hyperparameters["max_bits_per_feature"]
        nb_batch_ids = self.nb_batch_ids
        num_features = len(self.columns)

        for c_id, clause_lists in self.current_res.interpretability_clauses.items():
            for clause_idx, clause in enumerate(clause_lists):
                # Create a matrix: rows=features, cols=max_bits_per_feature
                
                matrix = np.zeros((num_features, max_bits_per_feature), dtype=int)
                for literal in clause:
                    match = re.match(r'^(¬| )x\s*(\d+)', literal)
                    # match = re.match(r'^(¬)?x\s*(\d+)', literal)
                    if match:
                        neg, idx = match.groups()
                        idx = int(idx)
                        if idx >= nb_batch_ids:
                            idx -= nb_batch_ids
                            feature = idx // max_bits_per_feature
                            bit = idx % max_bits_per_feature
                            if neg == '¬':
                                matrix[feature, bit] = -1  # Red for negative
                            else:
                                matrix[feature, bit] = 1   # Green for positive
                # matrix = matrix.T
                cmap = ListedColormap(['red', 'white', 'green'])
                bounds = [-1.5, -0.5, 0.5, 1.5]
                norm = BoundaryNorm(bounds, cmap.N)
                plt.imshow(matrix, aspect='auto', cmap=cmap, norm=norm)
                plt.ylabel("Features")
                plt.xlabel("Thresholds")
                plt.title(f"Class {c_id} - Clause {clause_idx}")
                plt.colorbar(ticks=[-1, 0, 1], label='Literal', format=lambda x, _: {1: 'x_i', 0: 'None', -1: '¬x_i'}.get(x, ''))
                plt.yticks(np.arange(num_features), self.columns)
                plt.xticks(np.arange(max_bits_per_feature), [fr"$\tau${b}" for b in range(max_bits_per_feature)])
                plt.show()
    
    def plot_batch_ids_by_class(self, plot_zeroes: bool = False):
        """
        For each class, plot the count of 1s and 0s for each of the first nb_batch_ids features in X.
        """

        for class_name, idx in self.current_res.ids_by_class.items():
            cls_batch_ids = [self.mhe_batch_ids[i] for i in idx]
            ones = np.sum(cls_batch_ids, axis=0)
            nb_batch_ids = len(ones)
            if plot_zeroes:
                zeros = cls_batch_ids.shape[0] - ones
            x = np.arange(nb_batch_ids)
            width = 0.35
            plt.figure(figsize=(10, 5))
            if plot_zeroes:
                plt.bar(x - width/2, zeros, width, label='0s')
                plt.bar(x + width/2, ones, width, label='1s')
            else:
                plt.bar(x + width/2, ones, width)
            # plt.axes().set_xticks(x, x, rotation=45)
            if self.plot_titles:
                plt.title(f"Class {class_name} - Count of Batch Ids")
            plt.xlabel("Batch Index")
            plt.ylabel("Count")
            # plt.xticks(x, [f"Batch {i}" for i in range(nb_batch_ids)])
            # plt.legend()
            plt.show()

    def plot_batch_ids(self):
        """
        Plot the batch ids for each class on the same plot, using a different color for each class.
        """
        plt.figure(figsize=(10, 5))
        colors = plt.cm.get_cmap('tab10', len(self.current_res.ids_by_class))
        for idx, (class_name, cls_idx) in enumerate(self.current_res.ids_by_class.items()):
            cls_batch_ids = np.array([self.mhe_batch_ids[i] for i in cls_idx])
            nb_batch_ids = cls_batch_ids.shape[1]
            x = np.arange(nb_batch_ids)
            plt.bar(x, cls_batch_ids.sum(axis=0), alpha=0.7, label=str(class_name), color=colors(idx))
        plt.xlabel("Batch Index")
        plt.ylabel("Count")
        if self.plot_titles:
            plt.title("Batch Ids by Class")
        plt.legend()
        plt.show()

    def plot_epoch_by_class(self):
        """
        Plot the epoch by class for each class.
        """
        
        for class_name, ids in self.current_res.ids_by_class.items():
            epochs = self.raw_data.iloc[ids]["epoch"]
            plt.figure(figsize=(10, 5))
            plt.hist(epochs, bins=range(int(epochs.min()), int(epochs.max()) + 2), alpha=0.7, color='skyblue', edgecolor='black')
            if self.plot_titles:
                plt.title(f"Epoch distribution for class {class_name}")
            else:
                plt.title(f"Epoch distribution for class {class_name} - Run {self.run}")
            plt.xlabel("Epoch")
            plt.ylabel("Count")
            plt.show()

    def plot_epochs(self):
        plt.figure(figsize=(10, 5))
        for class_name, ids in self.current_res.ids_by_class.items():
            epochs = self.raw_data.iloc[ids]["epoch"]
            plt.hist(epochs, bins=range(int(epochs.min()), int(epochs.max()) + 2), alpha=0.5, label=str(class_name), edgecolor='black')
        if self.plot_titles:
            plt.title("Epoch distribution by class")
        else:
            print("Epoch distribution")
        plt.xlabel("Epoch")
        plt.ylabel("Count")
        plt.legend()
        plt.show()

    def plot_answer_types(self, log_scale: bool = False):
        fin_dataset = datasets.load_dataset("ibm-research/finqa", "en", split="train")

        answers = list(map(clean_answer, fin_dataset))
        answer_types = list(map(get_answer_formats, answers))

        batch_ids = np.array([np.where(np.array(mh) == 1)[0] for mh in self.mhe_batch_ids])

        for c, idx in self.current_res.ids_by_class.items():
            cls_batch_ids = batch_ids[idx].flatten()
            answer_types_for_batch = [answer_types[i] for i in cls_batch_ids]
            unique_types, counts = np.unique(answer_types_for_batch, return_counts=True)
            plt.figure(figsize=(8, 4))
            if log_scale:
                plt.yscale('log')
            plt.bar(unique_types, counts)
            plt.xlabel("Answer Types")
            plt.ylabel("Count")
            if self.plot_titles:
                plt.title(f"Answer Type Distribution for Cluster {c}")
            else:
                print(f"Answer Type Distribution for Cluster {c} - Run {self.run}")
            plt.show()

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
            new_obj["grouped_samples"][classes[c_id]] = obj["grouped_samples"][k].astype(np.int8)

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


    def get_ids_by_class(self, data):
        grouped_samples = {k: v.tolist() for k, v in self.grouped_samples.items()}
        items_by_class = { k: 0 for k in grouped_samples.keys() }
        not_associated = 0
        ids_by_class = { k: [] for k in grouped_samples.keys() }
        not_associated_ids = []
        X = data.tolist()
        for x in X:
            associated = False
            for k, v in grouped_samples.items():
                if x in v:
                    items_by_class[k] += 1
                    idx = X.index(x)
                    ids_by_class[k].append(idx)
                    # ids_by_class[k].append(X.index(x))
                    associated = True
                    break
            if not associated:
                not_associated_ids.append(X.index(x))
                not_associated += 1

        if self.verbose:
            print("\nItems by class:", items_by_class)
            print("Number of not associated samples:", not_associated)
            print("Not associated samples ids:", not_associated_ids)
            print("ids by class:", ids_by_class)
        return ids_by_class, not_associated_ids
    
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
                    
