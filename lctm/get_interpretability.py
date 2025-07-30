from tm_data.preprocessing import AugmentedBinarizer, Binarizer, MaxThresholdBinarizer, DataPreprocessor
from plots.utils import columns_cleaned, models_cleaned
from llm.utils import get_model_dir
from llm.data.dataset import clean_answer, get_answer_formats
from utils import mplstyle_file, mplplots_dir, width, height
import datasets
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pickle
from typing import Union, Optional, Dict
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from enum import Enum
import re

from textwrap import dedent

mpl.rcParams['text.usetex'] = True

import re
from pathlib import Path
from textwrap import dedent

# def generate_latex_for_clauses_matrix(fig_dir: Path):
#     """
#     Generate LaTeX code for clause matrix figures (3x2 per page).
#     Each fig label is formatted as: fig:example_{#example}-runs_{#runs}_clauses_matrix_class_{class}_clause_{clause}
#     """
#     files = sorted(fig_dir.rglob("clauses_matrix_class_*.pdf"))
#     latex_blocks = []
#     current_page = []

#     def extract_metadata(file_path: Path):
#         # m = re.search(r"example_(\d+)", str(file_path))
#         # example = m.group(1) if m else "X"

#         m = re.search(r"runs_(\d+)", str(file_path))
#         runs = m.group(1) if m else "Y"

#         m = re.search(r"clauses_matrix_class_(.+?)_clause_(\d+)\.pdf", file_path.name)
#         class_id, clause_id = m.groups() if m else ("C", "Z")

#         return runs, class_id, int(clause_id)

#     for idx, fig_path in enumerate(files):
#         runs, class_id, clause_idx = extract_metadata(fig_path)

#         # Define label
#         fig_label = f"fig:runs_{runs}_clauses_matrix_class_{class_id}_clause_{clause_idx}"

#         # Define caption
#         if clause_idx == 0:
#             caption = f"Class $\\{class_id}$ – Accepting Clause"
#         elif clause_idx == 1:
#             caption = f"Class $\\{class_id}$ – Refusing Clause"
#         else:
#             caption = f"Class $\\{class_id}$ – Clause {clause_idx}"
#         file_path = Path(*fig_path.parts[fig_path.parts.index('lctm_res'):])  # ('lctm_res', 'llm_name', 'run_i', ...)

#         # Subfigure block
#         subfig = dedent(f"""
#             \\begin{{subfigure}}{{0.5\\textwidth}}
#                 \\centering
#                 \\includegraphics[width=\\linewidth]{{figs/{file_path.as_posix()}}}
#                 \\caption{{{caption}}}
#                 \\label{{{fig_label}}}
#             \\end{{subfigure}}%
#         """).strip()

#         current_page.append(subfig)

#         # Output full figure every 6 plots (3×2)
#         if len(current_page) == 20 or idx == len(files) - 1:
#             block = "\\begin{figure}[H]\n\\centering\n" + "\n\\vspace{0.5em}\n".join(current_page) + "\n\\end{figure}\n\n\\clearpage"
#             latex_blocks.append(block)
#             current_page = []

#     full_latex = "\n".join(latex_blocks)

#     with fig_dir.joinpath("clauses_matrix.tex").open("w") as f:
#         f.write(full_latex)
#     return full_latex


def generate_latex_for_clauses_matrix(fig_dir: Path):
    """
    Generate LaTeX code for clause matrix figures with a single figure split over multiple pages using \ContinuedFloat.
    """
    files = sorted(fig_dir.rglob("clauses_matrix_class_*.pdf"))
    latex_blocks = []
    current_page = []

    def extract_metadata(file_path: Path):
        m = re.search(r"runs_(\d+)", str(file_path))
        runs = m.group(1) if m else "Y"
        m = re.search(r"clauses_matrix_class_(.+?)_clause_(\d+)\.pdf", file_path.name)
        class_id, clause_id = m.groups() if m else ("C", "Z")
        return runs, class_id, int(clause_id)

    for idx, fig_path in enumerate(files):
        runs, class_id, clause_idx = extract_metadata(fig_path)

        fig_label = f"fig:runs_{runs}_clauses_matrix_class_{class_id}_clause_{clause_idx}"

        if clause_idx == 0:
            caption = f"Class $\\{class_id}$ – Accepting Clause"
        elif clause_idx == 1:
            caption = f"Class $\\{class_id}$ – Refusing Clause"
        else:
            caption = f"Class $\\{class_id}$ – Clause {clause_idx}"

        file_path = Path(*fig_path.parts[fig_path.parts.index('lctm_res'):])

        subfig = dedent(f"""
            \\begin{{subfigure}}{{0.48\\textwidth}}
                \\centering
                \\includegraphics[width=\\linewidth]{{figs/{file_path.as_posix()}}}
                \\subcaption*{{{caption}}}
                \\label{{{fig_label}}}
            \\end{{subfigure}}%
        """).strip()

        current_page.append(subfig)

        # Output every 8 subfigures as one figure page
        if len(current_page) == 8 or idx == len(files) - 1:
            is_first = (len(latex_blocks) == 0)
            figure_env = "\\begin{figure}[p]"
            if not is_first:
                figure_env += "\n\\ContinuedFloat"

            caption_block = ""
            if is_first:
                caption_block = "\n\\caption{Clause matrix visualization across classes and clauses}\n\\label{fig:clause_matrix}"
            else:
                caption_block = "\n\\caption[]{Clause matrix visualization continued}"

            block = (
                f"{figure_env}\n\\centering\n"
                + "\n\\vspace{0.5em}\n".join(current_page)
                + caption_block
                + "\n\\end{figure}\n"
            )
            latex_blocks.append(block)
            current_page = []

    full_latex = "\n\n".join(latex_blocks)

    with fig_dir.joinpath("clauses_matrix.tex").open("w") as f:
        f.write(full_latex)

    return full_latex

def generate_latex_epoch_dist(fig_path: Path):
    latex_blocks = []
    # base_rel_root = fig_path.parts[fig_path.parts.index('lctm_res'):]  # ('lctm_res', 'llm_name', 'run_i', ...)

    parts = fig_path.parts
    lctm_index = parts.index('lctm_res')
    llm_name = parts[lctm_index + 1]
    run_name = parts[lctm_index + 2]

    relative_path = Path(*parts[lctm_index:])

    # for llm_dir in base_path.iterdir():
        # if llm_dir.is_dir():
    # for run_dir in sorted(llm_dir.glob("run_*")):
    # fig_path = run_dir / "epoch_distribution_by_class.pdf"
    if fig_path.exists():
        label = f"fig:{llm_name}_{run_name}_epoch_dist"
        caption = f"Epoch distribution by class for {models_cleaned[llm_name]}."
        block = f"""
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\linewidth]{{figs/{relative_path.as_posix()}}}
    \\caption{{{caption}}}
    \\label{{{label}}}
\\end{{figure}}
"""
    latex_blocks.append(block)

    # Print all blocks to insert into a LaTeX document
    latex_code = "\n".join(latex_blocks)
    with fig_path.parent.joinpath("epoch_distribution.tex").open("w") as f:
        f.write(latex_code)

class LCTMResults:
    """
    Class to handle the results of the LCTM model.
    """

    def __init__(self, llm_name: str, verbose: bool = False, plot_titles: bool = False, save_figs: bool = False):
        self.llm_name = llm_name
        self.save_figs = save_figs
        self.verbose = verbose
        self.plot_dir = Path(mplplots_dir).joinpath("lctm_res", llm_name)
        if not self.plot_dir.exists():
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = get_model_dir(model_name=llm_name, return_error=True)
        self.interpretability_clauses_dir = self.model_dir.joinpath("interpretability_clauses")
        self.num_folders = sum(1 for p in self.interpretability_clauses_dir.iterdir() if p.is_dir())
        if self.verbose:
            print(f"Number of folders in interpretability_clauses: {self.num_folders}")
        self.get_path = lambda run: f"{self.current_example}/{run}"
        self.plot_titles = plot_titles
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
        # Count the number of elements (folders) already in self.plot_dir
        # nb_elements = len([p for p in self.plot_dir.iterdir() if p.is_dir()])
        # plot_dir = self.plot_dir.joinpath(f"example_{nb_elements}")
        plot_dir = self.plot_dir # .joinpath(f"example_{nb_elements}")

        # if nb_elements == 0:
        #     plot_dir.mkdir(parents=True, exist_ok=True)
        
        # elif self.plot_dir.joinpath(f"example_{nb_elements-1}").exists():
        #     if len([p for p in self.plot_dir.joinpath(f"example_{nb_elements-1}").iterdir() if p.is_dir()]) == 0:
        #         plot_dir = self.plot_dir.joinpath(f"example_{nb_elements-1}")
        #     else:
        #         plot_dir = self.plot_dir.joinpath(f"example_{nb_elements}")
        # else:
        #     plot_dir = self.plot_dir.joinpath(f"example_{nb_elements}")
        
        if not plot_dir.exists():
            plot_dir.mkdir(parents=True, exist_ok=True)
        self.current_res = LCTMCurrentResults(
            llm_name=self.llm_name,
            folder=self.current_folder, 
            model_dir=self.model_dir, 
            verbose=self.verbose, 
            plot_titles=self.plot_titles,
            plot_dir=plot_dir,
            save_figs=self.save_figs
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
    def __init__(
            self, 
            llm_name: str,
            folder: Union[str, Path], 
            model_dir: Path, 
            verbose: bool = False, 
            plot_titles: bool = False, 
            plot_dir: Optional[Path] = None,
            save_figs: bool = False
        ):
        self.llm_name = llm_name
        self.base_plot_dir = plot_dir
        
        self.save_figs = save_figs
        self.runs = [ f for f in folder.iterdir() if f.is_file() and "run_" in f.name ]
        
        self.verbose = verbose
        if len(self.runs) > 0:
            self.run = self.runs[0] 
        else:
            if self.verbose:
                print(f"No runs found in {folder}.")
            self.run = None
        self.current = 0
        self.max_runs = len(self.runs)
        self.model_dir = model_dir
        self.plot_titles = plot_titles
        # print(f"Data folder: {self.data_folder}")
        hp_file = folder.joinpath("hyperparameters.pkl") if folder.joinpath("hyperparameters.pkl").exists() else model_dir.parent.joinpath("default_hyperparameters.pkl")
        
        with open(hp_file, 'rb') as file:
            self.hyperparameters: Dict = pickle.load(file)
        
        document = self.hyperparameters.get("document", "fetched_batch_data.csv")
        self.hyperparameters["drop_batch_ids"] = self.hyperparameters.get("drop_batch_ids", False)

        if self.verbose:
            print(f"Current folder: {folder}")
            print(f"Hyperparameters file: {hp_file}")
            print(f"Hyperparameters: {self.hyperparameters}")
        try:
            self.data_folder = self.model_dir.joinpath(document)
        # elif document == "epoch":
        #     self.data_folder = self.model_dir.joinpath("fetched_training_data.csv")
        except:
            raise ValueError(f"Document {document} not supported. Choose between 'fetched_batch_data.csv' and 'epoch' fetched_training_data.csv.")
        
        self._get_binarizer()
        self._prepare_data()  # Parse the binarizer representation from hyperparameters
        self.nb_batch_ids = self.hyperparameters["nb_batch_ids"] if not self.hyperparameters["drop_batch_ids"] else 0
        # Parse the binarizer representation from hyperparameters
    
    def _prepare_data(self):
        retrieve_mhe_batch_ids = self.hyperparameters.get("document", "fetched_batch_data.csv") == "fetched_batch_data.csv"
        data_preprocesor = DataPreprocessor(
            csv_path=self.data_folder,
            binarizer=self.binarizer,
            columns_to_drop=self.hyperparameters["columns_dropped"],
            drop_batch_ids=self.hyperparameters["drop_batch_ids"],
            hash_batch_ids=self.hyperparameters.get("hash_batch_ids", False),
            verbose=self.verbose,
            retrieve_mhe_batch_ids=retrieve_mhe_batch_ids
        )
        data_preprocessed = data_preprocesor.fit_transform(max_id=self.hyperparameters["num_samples"], return_columns=True)
        if retrieve_mhe_batch_ids:
            self.data, self.columns, self.mhe_batch_ids = data_preprocessed
        else:
            self.data, self.columns = data_preprocessed
            self.mhe_batch_ids = []
        self.nb_batch_ids = data_preprocesor.nb_batch_ids
        # print("self.nb_batch_ids", self.nb_batch_ids)
        # print("self.columns", self.columns)
        # print("self.columns_dropped", data_preprocesor.columns_dropped)
        # print("data.shape", self.data.shape)
        # for k in range(len(self.columns)):
        #     print("column:", self.columns[k], ", unique_values:", len(data_preprocesor.binarizer.unique_values[k]))
        self.features_by_column = [ len(data_preprocesor.binarizer.unique_values[k]) for k in range(len(self.columns))]
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
            if self.verbose:
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
        nb_elements = len([p for p in self.base_plot_dir.iterdir() if p.is_dir()])
        # if nb_elements == 0:
        #     self.plot_dir = self.base_plot_dir.joinpath(f"run_{0}")
        if nb_elements > 0 and len(list(self.base_plot_dir.joinpath(f"run_{nb_elements-1}").iterdir())) == 0:
            self.plot_dir = self.base_plot_dir.joinpath(f"run_{nb_elements-1}")
        else:
            self.plot_dir = self.base_plot_dir.joinpath(f"run_{nb_elements}")

        if not self.plot_dir.exists():
            self.plot_dir.mkdir(parents=True, exist_ok=True)

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
        binarizer = self.binarizer_repr == "MaxThresholdBinarizer"
        if isinstance(self.data, list):
            data = np.array(self.data)
        else:
            data = self.data
        max_bits_per_feature = self.hyperparameters["max_bits_per_feature"]
        nb_batch_ids = self.nb_batch_ids
        if self.hyperparameters.get("document", "fetched_batch_data.csv") == "fetched_batch_data.csv":
            assert (data.shape[1] - nb_batch_ids) % max_bits_per_feature == 0, f"The number of features binarized should be a multiple of max_bits_per_feature value. Got: {data.shape[1] - nb_batch_ids} and {max_bits_per_feature}"
            nb_features = (data.shape[1] - nb_batch_ids) // max_bits_per_feature
        else:
            assert sum(self.features_by_column) == data.shape[1], f"The sum of the number of features by column should be equal to the number of features binarized. Got {sum(self.features_by_column)} and {data.shape[1]}."
            nb_features = len(self.features_by_column)
        for feature in range(nb_features):
            nb_features_binarized = self.features_by_column[feature] 
            x = data[:,feature * nb_features_binarized + nb_batch_ids:(feature + 1) * nb_features_binarized + nb_batch_ids]
            x_t = np.argmax(x, axis=1, keepdims=True) if binarizer else np.sum(x, axis=1, keepdims=True)
            X_T.append(x_t)
        X_T = np.concatenate(X_T, axis=1)
        return X_T
 
    def plot_features_threshold(self, with_features_colors=False):
        data = self.get_threshold_index_by_feature() # if self.binarizer_repr == "MaxThresholdBinarizer" else self.get_max_threshold_by_feature()

        for c_id, idx in self.current_res.ids_by_class.items():
            plt.figure(figsize=(width, height))
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
            plt.style.use(mplstyle_file)
            if self.save_figs:
                plt.savefig(self.plot_dir.joinpath(f"features_threshold_class_{c_id}.pdf"), bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def plot_threshold_features(self, with_features_colors=False):
        data = self.get_threshold_index_by_feature() #  if self.binarizer_repr == "MaxThresholdBinarizer" else self.get_max_threshold_by_feature()

        for c_id, idx in self.current_res.ids_by_class.items():
            plt.figure(figsize=(width, height))
            labels = list(self.columns)
            y = np.arange(data.shape[1])
            plt.yticks(y, labels, rotation=0)
            if with_features_colors:
                for i in range(data.shape[1]):
                    plt.scatter(data[idx, i], [i] * len(idx), label=labels[i] if i < len(labels) else f"Feature {i}")
            else:
                for i in range(data.shape[1]):
                    plt.scatter(data[idx, i], [i] * len(idx))
            # print(f"Scatter plot of thresholds by feature for class {c_id} on run {self.run}")
            plt.ylabel("Features")
            plt.xlabel("Thresholds")
            plt.style.use(mplstyle_file)
            if self.save_figs:
                plt.savefig(self.plot_dir.joinpath(f"threshold_features_class_{c_id}.pdf"), bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def plot_clauses_matrix(self):
        """
        Plot a matrix for each class and clause, where rows are features, columns are bits (max_bits_per_feature),
        and cells are colored green if x_i is present (True), red if ¬x_i (False), and white if not present.
        """
        max_bits_per_feature = self.hyperparameters["max_bits_per_feature"]
        features_by_column = self.features_by_column
        nb_batch_ids = self.nb_batch_ids
        num_features = len(self.columns)
        columns = [ fr"{column}" for column in self.columns ]
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
                            for feature, nb_ft_col in enumerate(features_by_column):
                                if idx < nb_ft_col:
                                    bit = idx
                                    break
                                else:
                                    idx -= nb_ft_col
                            # feature = idx // max_bits_per_feature
                            # bit = idx % max_bits_per_feature
                            if neg == '¬':
                                matrix[feature, bit] = -1  # Red for negative
                            else:
                                matrix[feature, bit] = 1   # Green for positive
                # matrix = matrix.T
                cmap = ListedColormap(['red', 'white', 'green'])
                bounds = [-1.5, -0.5, 0.5, 1.5]
                norm = BoundaryNorm(bounds, cmap.N)
                plt.figure(figsize=(width, height))
                plt.imshow(matrix, aspect='auto', cmap=cmap, norm=norm)
                plt.ylabel("Features")
                plt.xlabel("Thresholds")
                if self.plot_titles:
                    plt.title(f"Class $\\{c_id}$ - Clause {clause_idx}")
                elif self.verbose:
                    print(f"Class {c_id} - Clause {clause_idx} - Run {self.run}")
                plt.colorbar(ticks=[-1, 0, 1], label='Literal', format=lambda x, _: {1: '$x_i$', 0: 'None', -1: '$\\neg x_i$'}.get(x, ''))
                plt.yticks(np.arange(num_features), columns)
                plt.xticks(np.arange(max_bits_per_feature), [fr"$\tau_{b}$" for b in range(max_bits_per_feature)])
                plt.style.use(mplstyle_file)
                if self.save_figs:
                    plt.savefig(self.plot_dir.joinpath(f"clauses_matrix_class_{c_id}_clause_{clause_idx}.pdf"), bbox_inches='tight')
                    
                    plt.close()
                else:
                    plt.show()
        if self.save_figs:
            generate_latex_for_clauses_matrix(self.plot_dir)
    
    def plot_batch_ids_by_class(self, plot_zeroes: bool = False):
        """
        For each class, plot the count of 1s and 0s for each of the first nb_batch_ids features in X.
        """
        if self.nb_batch_ids == 0:
            if self.verbose:
                print("No batch ids to plot. The data has no features or only batch ids.")
            return
        for class_name, idx in self.current_res.ids_by_class.items():
            cls_batch_ids = [self.mhe_batch_ids[i] for i in idx]
            ones = np.sum(cls_batch_ids, axis=0)
            nb_batch_ids = len(ones)
            if plot_zeroes:
                zeros = cls_batch_ids.shape[0] - ones
            x = np.arange(nb_batch_ids)
            width = 0.35
            plt.figure(figsize=(width, height))
            if plot_zeroes:
                plt.bar(x - width/2, zeros, width, label='0s')
                plt.bar(x + width/2, ones, width, label='1s')
            else:
                plt.bar(x + width/2, ones, width)
            # plt.axes().set_xticks(x, x, rotation=45)
            if self.plot_titles:
                plt.title(f"Class $\\{class_name}$ - Count of Batch Ids")
            else: 
                plt.title(f"Count of Batch Ids for Class $\\{class_name}$ - Run {self.run}")
            plt.xlabel("Batch Index")
            plt.ylabel("Count")
            # plt.xticks(x, [f"Batch {i}" for i in range(nb_batch_ids)])
            # plt.legend()
            plt.style.use(mplstyle_file)
            plt.show()

    def plot_batch_ids(self):
        """
        Plot the batch ids for each class on the same plot, using a different color for each class.
        """
        if self.nb_batch_ids == 0:
            print("No batch ids to plot. The data has no features or only batch ids.")
            return
        plt.figure(figsize=(width, height))
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
        else:
            print("Batch Ids by Class - Run", self.run)

        plt.legend()
        plt.style.use(mplstyle_file)
        plt.show()

    def plot_epoch_by_class(self):
        """
        Plot the epoch by class for each class.
        """
        for class_name, ids in self.current_res.ids_by_class.items():
            epochs = self.raw_data.iloc[ids]["epoch"]
            plt.figure(figsize=(width, height))
            plt.hist(epochs, bins=range(int(epochs.min()), int(epochs.max()) + 2), alpha=0.7, color='skyblue', edgecolor='none')
            if self.plot_titles:
                plt.title(f"Epoch distribution for class {class_name}")
            else:
                print(f"Epoch distribution for class {class_name} - Run {self.run}")
            plt.xlabel("Epoch")
            plt.ylabel("Count")
            plt.style.use(mplstyle_file)
            plt.show()

    def plot_epochs(self):

        if self.hyperparameters.get("document", "fetched_batch_data.csv") == "fetched_batch_data.csv":
            return
            # self._plot_accumulated_batchs()
        elif self.hyperparameters.get("document", "fetched_batch_data.csv") == "fetched_training_data.csv":
            self._plot_epochs()
        else:
            raise ValueError(f"Document {self.hyperparameters.get('document', 'fetched_batch_data.csv')} not supported. Choose between 'fetched_batch_data.csv' and 'fetch_training_data.csv'.")
    
    def get_class_name(self):
        if len(self.current_res.ids_by_class.keys()) > 24:
            plot_classes = lambda x: str(x)
        else:
            plot_classes = lambda x: f"$\\{x}$"
        return plot_classes
    
    # def _plot_epochs(self):
    #     plot_classes = self.get_class_name()
    #     num_classes = len(self.current_res.ids_by_class.keys())

    #     colors = plt.cm.get_cmap('tab20', num_classes)
    #     plt.figure(figsize=(width, height))
    #     for i, (class_name, ids) in enumerate(self.current_res.ids_by_class.items()):
    #         epochs = self.raw_data.iloc[ids]["epoch"]
    #         # if len(self.current_res.ids_by_class.keys()):
    #         plt.hist(epochs, bins=range(int(epochs.min()), int(epochs.max()) + 2), alpha=0.5, label=plot_classes(class_name), color=colors(i), edgecolor='none') 
    #         # else:
    #         #     plt.hist(epochs, bins=range(int(epochs.min()), int(epochs.max()) + 2), alpha=0.5, loc="", label=plot_classes(class_name), edgecolor='none') 
    #     if self.plot_titles:
    #         plt.title("Epoch distribution by class")
    #     else:
    #         print("Epoch distribution")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Count")

    #     plt.yticks([0.0, 1.0])
    #     # plt.ytick(["0", "1"])
    #     plt.legend(
    #         bbox_to_anchor=(1.02, 1), 
    #         ncol=2 if num_classes > 8 else 1,
    #         title="Classes",
    #         borderaxespad=0,
    #     )
    #     plt.style.use(mplstyle_file)
    #     plt.show()
    def _plot_epochs(self):
        plot_classes = self.get_class_name()
        num_classes = len(self.current_res.ids_by_class.keys())
        colors = plt.cm.get_cmap('viridis', num_classes)

        # Step 1: Compute mean epoch per class
        class_epoch_means = []
        for class_name, ids in self.current_res.ids_by_class.items():
            epochs = self.raw_data.iloc[ids]["epoch"]
            class_epoch_means.append((class_name, epochs.min(), ids))

        # Step 2: Sort by mean epoch
        class_epoch_means.sort(key=lambda x: x[1])  # Sort by mean

        plt.figure(figsize=(width, height))

        # Step 3: Plot in order of mean epoch
        for i, (class_name, _, ids) in enumerate(class_epoch_means):
            epochs = self.raw_data.iloc[ids]["epoch"]
            plt.hist(
                epochs,
                bins=range(int(epochs.min()), int(epochs.max()) + 2),
                # alpha=0.5,
                label=plot_classes(class_name),
                color=colors(i),
                edgecolor='none'
            )

        if self.plot_titles and self.verbose:
            plt.title("Epoch distribution by class")
        elif self.verbose:
            print("Epoch distribution")

        plt.xlabel("Epoch")
        plt.ylabel("Class Count")  # Updated label

        # Show only 0 and 1 on y-axis
        plt.yticks([0.0, 1.0])

        # Legend outside
        plt.legend(
            bbox_to_anchor=(1.02, 1),
            # loc="upper left",
            ncol=2 if num_classes > 8 else 1,
            title="Classes",
            borderaxespad=0,
        )

        # plt.tight_layout(rect=[0, 0, 0.82, 1])  # Leave room for legend
        plt.style.use(mplstyle_file)
        if self.save_figs:
            plot_path = self.plot_dir.joinpath("epoch_distribution_by_class.pdf")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            generate_latex_epoch_dist(fig_path=plot_path)
        else:
            plt.show()

    def plot_number_of_elements_by_class(self):
        """
        Plot the number of elements by class using self.current_res.ids_by_class.
        """
        if self.verbose:
            print("Plotting number of elements by class.")
        ids_by_class = self.current_res.ids_by_class

        plot_classes = self.get_class_name()
        class_names = list(map(plot_classes, ids_by_class.keys()))

        counts = [len(ids) for ids in ids_by_class.values()]

        num_classes = len(self.current_res.ids_by_class.keys())
        colors = plt.cm.get_cmap('viridis', num_classes)
        colors = [colors(i) for i in range(num_classes)]

        plt.figure(figsize=(width, height))
        plt.bar(class_names, counts, color=colors)
        plt.xlabel("Class")
        plt.ylabel("Number of Elements")
        if self.plot_titles:
            plt.title("Number of Elements per Class")
        else:
            print("Number of Elements per Class - Run", self.run)
        plt.xticks(rotation=45)
        plt.style.use(mplstyle_file)
        if self.save_figs:
            plt.savefig(self.plot_dir.joinpath("elements_per_class.pdf"), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    # def _plot_accumulated_batchs(self):
    #     # Plot the accumulation steps by class, with a secondary x-axis for epochs
    #     if self.nb_batch_ids == 0 and self.verbose:
    #         print("No batch ids to plot. The data has no features or only batch ids.")
    #         return

    #     fig, ax1 = plt.subplots(figsize=(width, height))

    #     # Get the epoch column from raw_data
    #     epochs = self.raw_data["epoch"].values if hasattr(self.raw_data, "values") else self.raw_data["epoch"]

    #     plot_classes = self.get_class_name()

    #     for class_name, ids in self.current_res.ids_by_class.items():
    #         # Sort indices by epoch for correct accumulation
    #         class_epochs = epochs[ids]
    #         sorted_idx = np.argsort(class_epochs)
    #         sorted_epochs = class_epochs[sorted_idx]
    #         steps = np.arange(1, len(sorted_epochs) + 1)
    #         ax1.step(steps, sorted_epochs, label=plot_classes(class_name))

    #     ax1.set_xlabel("Accumulation Step")
    #     ax1.set_ylabel("Epoch")
    #     if self.plot_titles:
    #         ax1.set_title("Accumulation Steps by Class with Epoch Mapping")
    #     else:
    #         if self.verbose:
    #             print("Accumulation Steps by Class with Epoch Mapping - Run", self.run)
    #     ax1.legend()

    #     def step_to_epoch(step):
    #         all_epochs = []
    #         for ids in self.current_res.ids_by_class.values():
    #             class_epochs = epochs[ids]
    #             all_epochs.extend(class_epochs)
    #         all_epochs = np.sort(np.array(all_epochs))

    #         if isinstance(step, np.ndarray):
    #             # Clip to valid index range
    #             step = np.round(step).astype(int)
    #             step = np.clip(step, 1, len(all_epochs)) - 1
    #             return all_epochs[step]
            
    #         # Scalar case
    #         step = int(round(step))
    #         step = min(max(1, step), len(all_epochs))  # Clamp to valid range
    #         return int(all_epochs[step - 1])
            
    #     def epoch_to_step(epoch):
    #         all_epochs = []
    #         for ids in self.current_res.ids_by_class.values():
    #             class_epochs = epochs[ids]
    #             all_epochs.extend(class_epochs)
    #         all_epochs = np.sort(np.array(all_epochs))

    #         if isinstance(epoch, np.ndarray):
    #             steps = np.searchsorted(all_epochs, epoch, side="left") + 1
    #             steps = np.clip(steps, 1, len(all_epochs))
    #             return steps

    #         step = np.searchsorted(all_epochs, epoch, side="left") + 1
    #         return min(max(1, step), len(all_epochs))


    #     ax2 = ax1.secondary_xaxis('top', functions=(step_to_epoch, epoch_to_step))
    #     ax2.set_xlabel("Epoch at Step")

    #     plt.style.use(mplstyle_file)
    #     if self.save_figs:
    #         plt.savefig(self.plot_dir.joinpath("accumulation_steps_by_class.pdf"), bbox_inches='tight')
    #         plt.close()
    #     else:
    #         plt.show()

    def _plot_nb_accumulated_batchs(self):
        # Plot a histogram of the number of accumulation steps per class
        if self.nb_batch_ids == 0 and self.verbose:
            print("No batch ids to plot. The data has no features or only batch ids.")
            return

        fig, ax = plt.subplots(figsize=(width, height))

        num_classes = len(self.current_res.ids_by_class.keys())
        colors = plt.cm.get_cmap('viridis', num_classes)

        # Count accumulation steps per class
        plot_classes = self.get_class_name()
        class_names = []
        counts = []

        for i, (class_name, ids) in enumerate(self.current_res.ids_by_class.items()):
            class_names.append(plot_classes(class_name))
            counts.append(len(ids))  # Number of steps for that class

        # Plot histogram
        ax.bar(class_names, counts, color=colors)
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Accumulation Steps")
        ax.set_title("Histogram of Accumulation Steps by Class" if self.plot_titles else None)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.style.use(mplstyle_file)

        if self.save_figs:
            plt.savefig(self.plot_dir.joinpath("accumulation_histogram_by_class.pdf"), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_nb_elements_per_class(self):
        """
        Plot the number of elements per class.
        """
        if self.verbose:
            print("Plotting number of elements per class.")
        plt.figure(figsize=(width, height))
        class_names = list(self.current_res.ids_by_class.keys())
        counts = [len(ids) for ids in self.current_res.ids_by_class.values()]
        plt.bar(class_names, counts, color='skyblue')
        plt.xlabel("Class")
        plt.ylabel("Number of Elements")
        if self.plot_titles:
            plt.title("Number of Elements per Class")
        else:
            print("Number of Elements per Class - Run", self.run)
        plt.xticks(rotation=45)
        plt.style.use(mplstyle_file)
        if self.save_figs:
            plt.savefig(self.plot_dir.joinpath("elements_per_class.pdf"), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_answer_types(self, log_scale: bool = False):
        fin_dataset = datasets.load_dataset("ibm-research/finqa", "en", split="train")
        if self.nb_batch_ids == 0:
            print("No batch ids to plot. The data has no features or only batch ids.")
            return
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
            plt.style.use(mplstyle_file)
            plt.show()

    def plot_interpretability_clauses_image(self):
        
        """
        Plot a matrix for each class and clause, where rows are features, columns are bits (max_bits_per_feature),
        and cells are colored green if x_i is present (True), red if ¬x_i (False), and white if not present.
        """
        max_bits_per_feature = self.hyperparameters["max_bits_per_feature"]
        features_by_column = self.features_by_column
        nb_batch_ids = self.nb_batch_ids
        num_features = len(self.columns)
        columns = [fr"{column}" for column in self.columns]
        for c_id, clause_lists in self.current_res.interpretability_clauses.items():
            for clause_idx, clause in enumerate(clause_lists):
            # Create a matrix: rows=features, cols=max_bits_per_feature
                matrix = np.zeros((num_features, max_bits_per_feature), dtype=int)
                for literal in clause:
                    # Only plot validation literals: assume they are marked with a special character, e.g. '*'
                    # If you have a different way to identify validation literals, adjust this check accordingly.
                    if '*' not in literal:
                        continue
                    match = re.match(r'^(?:¬| )?x\s*(\d+)', literal.replace('*', ''))
                    if match:
                        idx = int(match.group(1))
                    if idx >= nb_batch_ids:
                        idx -= nb_batch_ids
                        for feature, nb_ft_col in enumerate(features_by_column):
                            if idx < nb_ft_col:
                                bit = idx
                                break
                            else:
                                idx -= nb_ft_col
                        matrix[feature, bit] = 1  # Mark validation literal as present
                plt.figure(figsize=(width, height))
                plt.imshow(matrix, aspect='auto', cmap='gray', vmin=0, vmax=1)
                plt.ylabel("Features")
                plt.xlabel("Thresholds")
                if self.plot_titles:
                    plt.title(f"Class $\\{c_id}$ - Clause {clause_idx} (Validation Literals Only)")
                else:
                    print(f"Class {c_id} - Clause {clause_idx} - Run {self.run} (Validation Literals Only)")
                plt.yticks(np.arange(num_features), columns)
                plt.xticks(np.arange(max_bits_per_feature), [fr"$\tau_{b}$" for b in range(max_bits_per_feature)])
                plt.style.use(mplstyle_file)
                plt.show()

    def get_epochs_by_class(self):
        """
        Get the epochs for each class.
        """
        return self.current_res.ids_by_class


class LCTMResult:
    def __init__(self, path, data: np.ndarray, hyperparameters: dict = None, verbose: bool = False):
        print(f"Self.verbose: {verbose}")
        self.verbose = verbose
        self.path = path
        self.hyperparameters = hyperparameters
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        greek = ["alpha", "beta", "chi", "delta", "epsilon", "eta", "gamma", "iota", "kappa", "lambda", "mu", "nu", "omega", "phi", "pi", "psi", "rho", "sigma", "tau", "theta", "upsilon", "xi", "zeta"]
        if len(list(obj["interpretability_clauses"].keys())) > 24:
            classes = range(len(obj["grouped_samples"]) )
        else:
            classes = greek

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

        class_mins = []
        for k, v in self.ids_by_class.items():
            if len(v) > 0:
                min_id = np.min(v)
            else:
                return
            class_mins.append((k, min_id))

        # Sort by min ID
        class_mins.sort(key=lambda x: x[1])
        sorted_keys = [k for k, _ in class_mins]
        num_classes = len(sorted_keys)

        if num_classes > len(greek):
            new_names = list(range(1, num_classes + 1))
        else:
            # greek = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
            new_names = greek[:num_classes]
        rename_map = {old: new for old, new in zip(sorted_keys, new_names)}

        # Apply renaming
        sorted_obj = {"interpretability_clauses": {}, "grouped_samples": {}}
        for old_k, new_k in rename_map.items():
            sorted_obj["interpretability_clauses"][new_k] = new_obj["interpretability_clauses"][old_k]
            sorted_obj["grouped_samples"][new_k] = new_obj["grouped_samples"][old_k].astype(np.int8)

        for k, v in sorted_obj.items():
            setattr(self, k, v)

        self.ids_by_class: dict[str, list[int]] = {
            rename_map[old]: self.ids_by_class[old] for old in sorted_keys
        }
        

    # def __init__(self, path, data: np.ndarray, hyperparameters: dict = None, verbose: bool = False):
    #     self.verbose = verbose
    #     self.path = path
    #     self.hyperparameters = hyperparameters
    #     with open(path, 'rb') as file:
    #         obj = pickle.load(file)

    #     new_obj = { "interpretability_clauses": {}, "grouped_samples": {} }
    #     new_obj["interpretability_clauses"] = obj["interpretability_clauses"]
    #     new_obj["grouped_samples"] = {k: v.astype(np.int8) for k, v in obj["grouped_samples"].items()}

    #     new_obj["interpretability_clauses"] = self._delete_empty_strings_in_clauses(new_obj["interpretability_clauses"])
    #     new_obj["interpretability_clauses"] = self._sort_clauses_by_feature(new_obj["interpretability_clauses"])

    #     for k, v in new_obj.items():
    #         setattr(self, k, v)
        
    #     # Also rename and sort ids_by_class
    #     self.ids_by_class, self.unassociated_samples = self.get_ids_by_class(data)

    #     for k,v in self.ids_by_class.items():
    #         if len(v) == 0:
    #             return
    #     # Temporary holders for sorting
    #     print("Ids by class before sorting:", self.ids_by_class)
    #     class_mins = []
    #     for k, v in self.ids_by_class.items():
    #         min_id = np.min(v)
    #         class_mins.append((k, min_id))

    #     # Sort by min ID
    #     class_mins.sort(key=lambda x: x[1])
    #     sorted_keys = [k for k, _ in class_mins]
    #     num_classes = len(sorted_keys)

    #     # Choose new names based on class count
    #     if num_classes > 8:
    #         new_names = list(range(1, num_classes + 1))
    #     else:
    #         greek = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    #         new_names = greek[:num_classes]

    #     # Build remapping
    #     print("Sorted keys:", sorted_keys)
    #     rename_map = {old: new for old, new in zip(sorted_keys, new_names)}

    #     # Apply renaming
    #     new_obj = {"interpretability_clauses": {}, "grouped_samples": {}}
    #     for old_k, new_k in rename_map.items():
    #         new_obj["interpretability_clauses"][new_k] = obj["interpretability_clauses"][old_k]
    #         new_obj["grouped_samples"][new_k] = obj["grouped_samples"][old_k].astype(np.int8)

    #     if self.verbose:
    #         print("\nNumber of classes:", num_classes)

    #     # Clean and sort interpretability clauses
    #     # new_obj["interpretability_clauses"] = self._delete_empty_strings_in_clauses(new_obj["interpretability_clauses"])
    #     # new_obj["interpretability_clauses"] = self._sort_clauses_by_feature(new_obj["interpretability_clauses"])

    #     self._get_missing_clauses(new_obj["interpretability_clauses"])

    #     for k, v in new_obj.items():
    #         setattr(self, k, v)

    #     self.ids_by_class = {
    #         rename_map[old]: self.ids_by_class[old] for old in sorted_keys
    #     }

    
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

        if self.verbose:
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
                        if self.verbose:
                            print("No associated samples")
                            print("Break at clause:", clause, "with id:", c_id)
                        break
                sample_clauses_mapping[key].append(list(associated_samples))
                unassociated_samples -= associated_samples

        unassociated_samples = list(unassociated_samples)
        return unassociated_samples
    

    # def get_ids_by_class(self, data):
    #     grouped_samples = {k: v.tolist() for k, v in self.grouped_samples.items()}
    #     items_by_class = { k: 0 for k in grouped_samples.keys() }
    #     not_associated = 0
    #     ids_by_class = { k: [] for k in grouped_samples.keys() }
    #     not_associated_ids = []
    #     X = data.tolist()
    #     for x in X:
    #         associated = False
    #         for k, v in grouped_samples.items():
    #             if x in v:
    #                 items_by_class[k] += 1
    #                 idx = X.index(x)
    #                 ids_by_class[k].append(idx)
    #                 # ids_by_class[k].append(X.index(x))
    #                 associated = True
    #                 break
    #         if not associated:
    #             not_associated_ids.append(X.index(x))
    #             not_associated += 1

    #     if self.verbose:
    #         print("\nItems by class:", items_by_class)
    #         print("Number of not associated samples:", not_associated)
    #         print("Not associated samples ids:", not_associated_ids)
    #         print("ids by class:", ids_by_class)
    #     return ids_by_class, not_associated_ids
    
    def get_ids_by_class(self, data):
        grouped_samples = {k: [tuple(row) for row in v.tolist()] for k, v in self.grouped_samples.items()}
        items_by_class = {k: 0 for k in grouped_samples.keys()}
        not_associated = 0
        ids_by_class = {k: [] for k in grouped_samples.keys()}
        not_associated_ids = []
        X = [tuple(row) for row in data.tolist()]
        assigned = [False] * len(X)

        for k, v in grouped_samples.items():
            for i, x in enumerate(X):
                if not assigned[i] and x in v:
                    ids_by_class[k].append(i)
                    items_by_class[k] += 1
                    assigned[i] = True  # Mark as assigned to avoid double assignment

        for i, was_assigned in enumerate(assigned):
            if not was_assigned:
                not_associated_ids.append(i)
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
            if len(duplicates) < data.shape[0] and self.verbose:
                print(f"Number of duplicate rows: {data.shape[0] - len(duplicates)}")
            elif self.verbose:
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
                    
