from utils import mplstyle_file, mplplots_dir, width, height
from plots.utils import columns_cleaned, models_cleaned
from llm.utils import get_model_dir
from llm.data.dataset import clean_answer, get_answer_formats

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datasets
from pathlib import Path


width, height = width / 1.2, height / 1.2

class TrainingResult:
    def __init__(self, model_name: str, verbose: bool = False, save: bool = False):
        self.model_name = model_name
        self.model_dir = get_model_dir(model_name, return_error=True)
        self.training_batch_data = pd.read_csv(self.model_dir.joinpath("fetched_batch_data.csv")).iloc[1:,]
        self.training_epoch_data = pd.read_csv(self.model_dir.joinpath("fetched_training_data.csv")).iloc[1:,]
        self.verbose = verbose
        self.default_outliers = ["$<-4\sigma$", "$>4\sigma$"]
        # self.default_outliers = ["<-4\sigma", ">4\sigma"]
        self.save = save
        self.dir = Path(mplplots_dir).joinpath("llm_monitoring").joinpath(self.model_name)
        if not self.dir.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
        self.files_to_save = []

    def __repr__(self):
        return f"TrainingResult(model_name={self.model_name}, training_time={self.training_time}, loss={self.loss})"
    
    def _baseplot(self, title: str, grid: bool = True, tight_layout: bool = True, plot: bool = True, path: str = None, file_name: str = None):
        path_dir = self.dir if path is None else self.dir.joinpath(path)
        file_name = file_name if file_name else title
        if not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)
        plt.grid(grid)
        if tight_layout:
            plt.tight_layout()

        if self.verbose:
            plt.title(title)
        if plot:
            if self.save:
                plt.savefig(path_dir.joinpath(f'{file_name}.pdf'), dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.close()
        self.files_to_save.append({path_dir.joinpath(f'{file_name}.pdf'): title})
        

    def _prepare_plot(self, xlabel: str, ylabel: str, figsize=(width, height)):
        plt.figure(figsize=figsize)
        xlabel = columns_cleaned.get(xlabel, xlabel)
        ylabel = columns_cleaned.get(ylabel, ylabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def plot_grad_data(self, show_outliers: bool = False, plot: bool = True):
        m_name = models_cleaned.get(self.model_name, self.model_name)
        columns_to_plot = self.training_batch_data.columns.tolist()
        columns_to_not_plot = ["batch_ids", "epoch", "train_loss", "val_loss", "accuracy_train", "confidence_train", "recall_train", "precision_train", "f1_train", "perplexity_train",]
        for col in columns_to_not_plot:
            if col in columns_to_plot:
                columns_to_plot.remove(col)
        columns_to_plot = ["grad_mean", "grad_abs_mean", "grad_median", "grad_std", "grad_max", "grad_min", "grad_dir", "grad_cos_dist", "grad_noise_scale"] 
        for col in columns_to_plot:
            c_clean = columns_cleaned.get(col, col)
            title = f'{c_clean} for {m_name}'
            self._prepare_plot('Step', col)
            plt.plot(self.training_batch_data[col])
            if show_outliers:
                outliers = self.training_batch_data[col].astype(float).abs() > 3 * self.training_batch_data[col].astype(float).std()
                plt.scatter(np.where(outliers)[0], self.training_batch_data[col][outliers], color='red', label='Outliers')
            self._baseplot(title, plot=plot)
    
    # def plot_feature(self, feature: str, plot: bool = True, figsize=(width, height), label: str = None):
    #     """
    #     Plot a specific feature from the training batch data.
    #     """
    #     if feature not in self.training_batch_data.columns:
    #         raise ValueError(f"Feature '{feature}' not found in training batch data.")
        
    #     feature_cleaned = columns_cleaned.get(feature, feature)
    #     title = f'{feature_cleaned} for {self.model_name}'
    #     if plot:
    #         xlabel = 'Step'
    #         ylabel = feature_cleaned
    #         self._prepare_plot(xlabel, ylabel, figsize=figsize)
    #     if not label is None:
    #         plt.plot(self.training_batch_data[feature].astype(float), label=label, alpha=0.5)
    #     else: 
    #         plt.plot(self.training_batch_data[feature].astype(float))
    #     self._baseplot(title, plot=plot)
    def plot_feature(self, feature: str, plot: bool = True, figsize=(6, 4), label: str = None):
        if feature not in self.training_batch_data.columns or feature == "ece":
            raise ValueError(f"Feature '{feature}' not found in training batch data.")

        feature_cleaned = columns_cleaned.get(feature, feature)
        title = f'{feature_cleaned} for {self.model_name}'

        y = self.training_batch_data[feature].astype(float).values
        x = np.arange(len(y))  # Explicit x-axis: step index
        
        if plot:
            xlabel = 'Accumulation Step'
            ylabel = feature_cleaned
            self._prepare_plot(xlabel, ylabel, figsize=figsize)

        if label is not None:
            plt.plot(x, y, label=label, alpha=0.5)
        else:
            plt.plot(x, y)
        
        self._baseplot(title, plot=plot)
    
    def get_feature_data(self, feature: str, normalize: bool = False):
        """
        Return x (step or normalized %), y (feature values).
        """
        if feature not in self.training_batch_data.columns:
            raise ValueError(f"Feature '{feature}' not found in training batch data.")

        y = self.training_batch_data[feature].astype(float).values
        if normalize:
            x = np.linspace(0, 100, len(y))  # 0–100% of training
        elif "step" in self.training_batch_data.columns:
            x = self.training_batch_data["step"].values
        else:
            x = np.arange(len(y))
        return x, y
    
    def get_feature_data_by_epoch(self, feature: str, normalize: bool = False):
        """
        Return x (step or normalized %), y (feature values).
        """
        # if feature not in self.training_epoch_data.columns + ["ece"]:
        #     raise ValueError(f"Feature '{feature}' not found in training epoch data.")
        if feature == "ece":
            y = self.compute_excepted_calibration_error()
        else:
            y = self.training_epoch_data[feature].astype(float).values
        if normalize:
            x = np.linspace(0, 100, len(y))  # 0–100% of training
        elif "step" in self.training_batch_data.columns:
            x = self.training_batch_data["step"].values
        else:
            x = np.arange(len(y))
        return x, y

    def plot_epoch_data(self):
        columns_to_not_plot = ["batch_ids", "batch_size", "epoch", "train_loss", "val_loss", "var_train_loss", "var_val_loss", "mean_lr"]
        metrics = ["accuracy", "confidence", "recall", "precision", "f1", "perplexity"]
        m_name = models_cleaned.get(self.model_name, self.model_name)
        for col in metrics:
            columns_to_not_plot.append(f'{col}_train')
            columns_to_not_plot.append(f'{col}_val')

        for col in self.training_epoch_data.columns:
            
            if not col in columns_to_not_plot:
                col_cleaned = columns_cleaned.get(col, col)
                title = f'{col_cleaned} for {m_name} by Epoch'
                self._prepare_plot('Epoch', col, figsize=(width, height))
                plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data[col], label=col)
                self._baseplot(title)

    def plot_couple_data(self):
        m_list = ["accuracy", "recall", "precision", "f1", "confidence", "perplexity"]
        m_name = models_cleaned.get(self.model_name, self.model_name)
        for metric in m_list:
            m_cleaned = columns_cleaned.get(metric, metric)
            m_train = columns_cleaned.get(f'{metric}_train', metric)
            m_val = columns_cleaned.get(f'{metric}_val', metric)
            title = f'{m_cleaned} {m_name} by Epoch'
            self._prepare_plot('Epoch', metric, figsize=(width, height))
            plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data[f'{metric}_train'], label="Train")
            plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data[f'{metric}_val'], label="Validation")
            # plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data[f'{metric}_train'], label=m_train)
            # plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data[f'{metric}_val'], label=m_val)
            plt.legend()
            self._baseplot(title)

        title = f'{m_name} Losses by Epoch'
        self._prepare_plot('Epoch', 'Losses Value', figsize=(width, height))
        plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data['train_loss'], label='Train')
        plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data['val_loss'], label='Validation')
        plt.legend()
        self._baseplot(title)

    def get_seasonality(self, data_type: str = "batch", top_seasonalities: int = 3):
        """
        """
        if data_type not in ["batch", "epoch"]:
            raise ValueError("data_type must be either 'batch' or 'epoch'")
        tr_data = self.training_batch_data if data_type == "batch" else self.training_epoch_data

        features_and_time_steps = {"grad_abs_mean": 0, "grad_mean": 0, "grad_median": 0, "grad_std": 0, "grad_dir": 0, "grad_cos_dist": 0}
        for metric, time_step in features_and_time_steps.items():
            
            data = tr_data[metric].astype(float).values[time_step:]
            n = len(data)
            t = np.arange(n)
            # Fourier Transform
            fft_vals = np.fft.fft(data)
            fft_freq = np.fft.fftfreq(n)
            # Get top 3 frequencies (excluding the zero frequency)
            idx = np.argsort(np.abs(fft_vals[1:n//2]))[::-1][:top_seasonalities] + 1
            important_freqs = fft_freq[idx]
            important_amps = fft_vals[idx]
            # Reconstruct signal using only the most important frequencies
            reconstructed = np.zeros(n, dtype=complex)
            for i, freq in zip(idx, important_freqs):
                reconstructed += fft_vals[i] * np.exp(2j * np.pi * freq * t)
                reconstructed += fft_vals[-i] * np.exp(2j * np.pi * -freq * t)
            reconstructed = reconstructed.real / n + data.mean()
            # Plot original and reconstructed
            title = f'Seasonality in {metric} for {self.model_name}'
            self._prepare_plot('Step', metric, figsize=(width, height))
            plt.plot(t, data, label=f'Original {metric}')
            plt.plot(t, reconstructed, label=f'Reconstructed {metric} (top {top_seasonalities} seasonalities)')
            self._baseplot(title)


    def get_batch_ids_by_distribution_range(self, rolling_window: int = 0, features: list = None, epoch_ids: list = []):
        """
        Get batch ids by distribution range.
        """
        if features is None:
            features = ["grad_abs_mean", "grad_mean", "grad_median", "grad_std", "grad_dir", "grad_cos_dist", "var_train_loss"]
        if rolling_window > 0:
            # Apply rolling mean and std if rolling_window is specified
            mean = self.training_batch_data[features].astype(float).rolling(window=rolling_window, min_periods=1).mean().iloc[rolling_window:]
            std = self.training_batch_data[features].astype(float).rolling(window=rolling_window, min_periods=1).std().iloc[rolling_window:]
        else:
            mean = self.training_batch_data[features].astype(float).mean(axis=0)
            std = self.training_batch_data[features].astype(float).std(axis=0)

        distribution_ranges = {
            "$<-4\sigma$": lambda x, m, s: x < m - 4 * s,
            "$<-3\sigma$": lambda x, m, s: x < m - 3 * s,
            "$-4\sigma$-$-3\sigma$": lambda x, m, s: (x >= m - 4 * s) & (x < m - 3 * s),
            "$-3\sigma$-$-2\sigma$": lambda x, m, s: (x >= m - 3 * s) & (x < m - 2 * s),
            "$-2\sigma-$-\sigma$": lambda x, m, s: (x >= m - 2 * s) & (x < m - 1 * s),
            "$-\sigma-$\sigma$": lambda x, m, s: (x >= m - 1 * s) & (x <= m + 1 * s),
            "$\sigma-$2\sigma$": lambda x, m, s: (x > m + 1 * s) & (x <= m + 2 * s),
            "$2\sigma-$3\sigma$": lambda x, m, s: (x > m + 2 * s) & (x <= m + 3 * s),
            "$3\sigma-$4\sigma$": lambda x, m, s: (x > m + 3 * s) & (x <= m + 4 * s),
            "$>3\sigma$": lambda x, m, s: x > m + 3 * s,
            "$>4\sigma$": lambda x, m, s: x > m + 4 * s,
        }
        # distribution_ranges = {
        #     "<-3\sigma": lambda x, m, s: x < m - 3 * s,
        #     "-3\sigma--2\sigma": lambda x, m, s: (x >= m - 3 * s) & (x < m - 2 * s),
        #     "-2\sigma--1\sigma": lambda x, m, s: (x  >= m - 2 * s) & (x < m - 1 * s),
        #     "-1\sigma-1\sigma": lambda x, m, s: (x >= m - 1 * s) & (x <= m + 1 * s),
        #     "1\sigma-2\sigma": lambda x, m, s: (x > m + 1 * s) & (x <= m + 2 * s),
        #     "2\sigma-3\sigma": lambda x, m, s: (x > m + 2 * s) & (x <= m + 3 * s),
        #     ">3\sigma": lambda x, m, s: x > m + 3 * s,
        # }
        batch_ids = { feature: {shift: [] for shift in distribution_ranges} for feature in features }
        if len(epoch_ids) > 0:
            # Filter the training batch data by epoch_ids if provided
            # self.training_batch_data = self.training_batch_data[self.training_batch_data['epoch'].isin(epoch_ids)]
            features_to_filter = features + ['epoch']
        else:
            features_to_filter = features
        data = self.training_batch_data[features_to_filter].astype(float).iloc[rolling_window:]
        # Convert string representations of lists into actual lists of int
        batch_ids_arr = self.training_batch_data['batch_ids'].apply(lambda x: list(map(int, x.strip('[]').split(',')))).values
        for feature in features:
            for shift, cond in distribution_ranges.items():
                if len(epoch_ids) > 0:
                    # Filter the data by epoch_ids
                    mask = cond(data[feature], mean[feature], std[feature]) & data['epoch'].isin(epoch_ids)
                else:
                    mask = cond(data[feature], mean[feature], std[feature]).values
                indices = np.where(mask)
                batch_ids[feature][shift].extend(batch_ids_arr[indices])

        for feature in features:
            for shift in distribution_ranges:
            # Count repetitions of the different ids
                unique, counts = np.unique(batch_ids[feature][shift], return_counts=True)
                batch_ids[feature][shift] = dict(zip(unique, counts))
        return batch_ids
    
    def plot_batch_ids_by_distribution_range(self, log_scale: bool = False, outliers: list = []):
        """
        Plot batch ids by distribution range.
        """
        outliers = outliers if len(outliers) > 0 else self.default_outliers
        batch_ids = self.get_batch_ids_by_distribution_range()
        print("batch_ids", batch_ids)
        for feature, shifts in batch_ids.items():
            title = f'Batch IDs by Distribution Range for {feature}'
            self._prepare_plot('Batch IDs', 'Count', figsize=(width, height))
            for shift, ids in shifts.items():
                if len(ids) > 0 and shift in outliers:
                    plt.bar(ids.keys(), ids.values(), label=shift)
            
            if log_scale:
                plt.yscale('log')
            self._baseplot(title)

    def plot_answer_types(self, log_scale: bool = False, rolling_window: int = 0, outliers: list = []):
        fin_dataset = datasets.load_dataset("ibm-research/finqa", "en", split="train")

        answers = list(map(clean_answer, fin_dataset))
        answer_types = list(map(get_answer_formats, answers))

        # batch_ids = np.array([np.where(np.array(mh) == 1)[0] for mh in self.mhe_batch_ids])
        batch_ids = self.get_batch_ids_by_distribution_range(rolling_window) # { feature: {shift: [] for shift in distribution_ranges } for feature in features }

        all_unique_types, all_counts = np.unique(answer_types, return_counts=True)

        bar_width = 0.1
        outliers = outliers if len(outliers) > 0 else self.default_outliers
        for feature, shifts in batch_ids.items():
            f_cleaned = columns_cleaned.get(feature, feature)
            title = f'Answer Types for {f_cleaned}'
            self._prepare_plot('Answer Types', 'Count (\%)', figsize=(width, height))
            if log_scale:
                plt.yscale('log')
            x = np.arange(len(all_unique_types))
            b_width = 0
            for c, idx in shifts.items():
                if len(idx) > 0 and c in outliers:
                    cls_batch_ids = idx
                    answer_types_for_batch = [answer_types[i] for i in cls_batch_ids]
                    unique_types, counts = np.unique(answer_types_for_batch, return_counts=True)
                    unique_types_idx = [np.where(all_unique_types == ut)[0][0] for ut in unique_types]
                    bar = counts / all_counts[unique_types_idx] * 100 if len(counts) > 0 else 0.
                    plt.bar(np.array(unique_types_idx) + bar_width * b_width, bar, width=bar_width, label=f'Range {c}', alpha=0.7)
                    b_width += 1
            # plt.bar(x, all_counts, width=bar_width, label='All Answer Types', alpha=0.7)
            if b_width == 0:
                print(f"No outliers found for {feature} in the specified range.")
                plt.close()
                continue
            plt.xticks(x + bar_width * (b_width - 1) / 2 , all_unique_types)
            # plt.xlabel("Answer Types")
            # plt.ylabel("Count (%)")
            plt.legend()
            # if self.plot_titles:
            self._baseplot(title, tight_layout=False)

    def plot_answer_types_outliers(self, log_scale: bool = False, rolling_window: int = 0, outliers: list = []):
        """
        Plot answer types for outliers.
        """
        outliers = outliers if len(outliers) > 0 else self.default_outliers
        batch_ids = self.get_batch_ids_by_distribution_range(rolling_window)
        for feature, shifts in batch_ids.items():
            title = f'Batch IDs by Distribution Range for {feature}'
            self._prepare_plot('Batch IDs', 'Count', figsize=(width, height))
            if log_scale:
                plt.yscale('log')
            for shift, ids in shifts.items() and shift in outliers:
                if len(ids) > 0:
                    plt.bar(ids.keys(), ids.values(), label=shift)
            self._baseplot(title)

    def plot_excepted_calibration_error(self, plot: bool = True):
        
        model_name = models_cleaned[self.model_name]
        ece_scores = self.compute_excepted_calibration_error(self, plot)
        epochs = np.arange(len(ece_scores))
        self._prepare_plot('Epoch', 'Expected Calibration Error', figsize=(width, height))
        plt.plot(epochs, ece_scores, label=model_name, marker='o')
        self._baseplot(f'Expected Calibration Error for {model_name}')
        if plot:
            plt.show()
        else:
            return ece_scores
    
    def compute_excepted_calibration_error(self):
        fin_dataset = datasets.load_dataset("ibm-research/finqa", "en", split="train")
        n = len(fin_dataset)
        data = self.training_batch_data[["accuracy_train","confidence_train","epoch"]]
        first_bin = self.training_batch_data['batch_ids'].apply(lambda x: list(map(int, x.strip('[]').split(',')))).values[0]
        bin_size = len(first_bin)
        ece_scores = data.groupby('epoch').apply(
            lambda x: sum(np.abs(x['accuracy_train'] - x['confidence_train'])) * bin_size / len(x)
        )
        self.ece_scores = ece_scores
        return ece_scores

    def plot_grad_histograms(self, bins: int = 50, log_scale: bool = False, plot=True):
        """
        Plot histogram of the specified gradient feature.
        """
        features = ["grad_mean", "grad_abs_mean", "grad_max", "grad_min", "grad_dir", "grad_cos_dist", "grad_noise_scale", "var_train_loss"] 
        for feature in features:
            self.plot_grad_histogram(feature, bins=bins, log_scale=log_scale, plot=plot)
            # ft = columns_cleaned.get(feature, feature)
            # m_name = models_cleaned.get(self.model_name, self.model_name)
            # title = f'{m_name} {ft} Histogram'
            # self._prepare_plot(feature, 'Frequency', figsize=(width, height))
            # plt.hist(self.training_batch_data[feature].astype(float), bins=bins, alpha=0.7, density=not plot)
            # if log_scale:
            #     plt.yscale('log')
            # self._baseplot(title, plot=plot)

    def plot_grad_histogram(
            self, 
            feature: str, 
            bins: int = 50, 
            log_scale: bool = False, 
            plot=True, 
            label: str = None, 
            density: bool = False, 
            percentages: bool = False, 
            alpha: float = 0.7,
            color: str = 'blue',
        ):
        """
        Plot histogram of the specified gradient feature.
        """
        # features = ["grad_mean", "grad_abs_mean", "grad_max", "grad_min", "grad_dir", "grad_cos_dist", "grad_noise_scale", "var_train_loss"] 
        # for feature in features:
        ft = columns_cleaned.get(feature, feature)
        m_name = models_cleaned.get(self.model_name, self.model_name)
        title = f'{m_name} {ft} Histogram'
        if plot:
            self._prepare_plot(feature, 'Frequency', figsize=(width, height))
        if percentages: 
            # Convert to percentages
            counts, _ = np.histogram(self.training_batch_data[feature].astype(float), bins=bins, density=density)
            counts = counts / counts.sum() * 100
            if label is not None:
                plt.bar(np.arange(len(counts)), counts, alpha=alpha, label=label, color=color)
            else:
                plt.bar(np.arange(len(counts)), counts, alpha=alpha)
        elif label is not None:
            plt.hist(self.training_batch_data[feature].astype(float), bins=bins, alpha=alpha, density=density, label=label, color=color)
        else:
            plt.hist(self.training_batch_data[feature].astype(float), bins=bins, alpha=alpha, density=density, color=color)
        if log_scale:
            plt.yscale('log')
        # if feature == "grad_noise_scale":
        #     plt.xlim(0, 1)
        # if feature == "grad_max" and not plot:
        #     # plt.xlim(0, 5000)
        #     plt.yscale('log')
        # if feature == "grad_min" and not plot:
        #     # plt.xlim(-5000, 0)
        #     plt.yscale('log')
        self._baseplot(title, plot=plot)

    def plot_answer_types_by_features_outliers_for_epochs(
            self,
            features: list = ["grad_mean", "grad_abs_mean", "grad_median", "grad_std", "grad_dir", "grad_cos_dist", "var_train_loss"],
            outliers: list = [],
            epochs: list = [],
            rolling_window: int = 0,
            log_scale: bool = False,
            percentages: bool = False,
            bins: int = 50,
            alpha: float = 0.7,
            color: str = 'blue',
        ):
        """
        Plot answer types by features outliers for epochs.
        """
        outliers = outliers if len(outliers) > 0 else self.default_outliers
        if len(epochs) == 0:
            epochs = self.training_epoch_data['epoch'].unique()
        if len(features) == 0:
            features = ["grad_mean", "grad_abs_mean", "grad_median", "grad_std", "grad_dir", "grad_cos_dist", "var_train_loss"]
        
        fin_dataset = datasets.load_dataset("ibm-research/finqa", "en", split="train")
        answers = list(map(clean_answer, fin_dataset))
        answer_types = list(map(get_answer_formats, answers))

        for feature in features:
            feature_cleaned = columns_cleaned.get(feature, feature)
            title = f'Answer Types by {feature_cleaned} Outliers for Epochs'
            self._prepare_plot('Epoch', 'Count (%)', figsize=(width, height))
            if log_scale:
                plt.yscale('log')
            for epoch in epochs:
                batch_ids = self.get_batch_ids_by_distribution_range(rolling_window)[feature]
                for shift, ids in batch_ids.items():
                    if len(ids) > 0 and shift in outliers:
                        cls_batch_ids = ids
                        answer_types_for_batch = [answer_types[i] for i in cls_batch_ids]
                        unique_types, counts = np.unique(answer_types_for_batch, return_counts=True)
                        all_unique_types, all_counts = np.unique(answer_types, return_counts=True)
                        unique_types_idx = [np.where(all_unique_types == ut)[0][0] for ut in unique_types]
                        bar = counts / all_counts[unique_types_idx] * 100 if len(counts) > 0 else 0.
                        plt.bar(np.array(unique_types_idx), bar, width=0.1, label=f'Epoch {epoch} {shift}', alpha=alpha)
            plt.xticks(np.arange(len(all_unique_types)), all_unique_types)
            plt.xlabel("Answer Types")
            plt.ylabel("Count (%)")
            plt.legend()
            self._baseplot(title, plot=True) 

    def plot_answer_types_by_features_outliers_for_batch_ids(
            self,
            features: list = ["grad_mean", "grad_abs_mean", "grad_median", "grad_std", "grad_dir", "grad_cos_dist", "var_train_loss"],
            outliers: list = [],
            epochs_by_class: dict[str, list] = {},
            rolling_window: int = 0,
            log_scale: bool = False,
            bins: int = 50,
            alpha: float = 0.7,
            color: str = 'blue',
        ):
            """
            Plot answer types by features outliers for batch IDs.
            """
            if len(epochs_by_class.keys()) == 0:
                return "No epochs found for the specified classes."
            
            fin_dataset = datasets.load_dataset("ibm-research/finqa", "en", split="train")
            answers = list(map(clean_answer, fin_dataset))
            answer_types = list(map(get_answer_formats, answers))
            all_unique_types, all_counts = np.unique(answer_types, return_counts=True)
                                
            outliers = outliers if len(outliers) > 0 else self.default_outliers
            
            if len(features) == 0:
                features = ["grad_mean", "grad_abs_mean", "grad_median", "grad_std", "grad_dir", "grad_cos_dist", "var_train_loss"]
            for class_name, epochs in epochs_by_class.items():
                if len(epochs) == 0:
                    print(f"No epochs found for class {class_name}. Skipping.")
                    continue
                
                for feature in features:
                    batch_ids = self.get_batch_ids_by_distribution_range(rolling_window, features=[feature], epoch_ids=epochs)

                    feature_cleaned = columns_cleaned.get(feature, feature)
                    title = f'{feature_cleaned} Outliers by Answer Types for Class {class_name}'
                    self._prepare_plot('Answer Types', 'Count (\%)', figsize=(width, height))
                    if log_scale:
                        plt.yscale('log')
                    for shift, ids in batch_ids.items():
                        if len(ids) > 0 and shift in outliers:
                            cls_batch_ids = ids
                            answer_types_for_batch = [answer_types[i] for i in cls_batch_ids]
                            unique_types, counts = np.unique(answer_types_for_batch, return_counts=True)
                            unique_types_idx = [np.where(all_unique_types == ut)[0][0] for ut in unique_types]
                            bar = counts / all_counts[unique_types_idx] * 100 if len(counts) > 0 else 0.
                            plt.bar(np.array(unique_types_idx), bar, width=0.1, label=f'{shift}', alpha=alpha)
                    plt.xticks(np.arange(len(all_unique_types)), all_unique_types)
                    plt.xlabel("Answer Types")
                    plt.ylabel("Count (%)")
                    plt.legend()
                    self._baseplot(title, plot=True)

    def plot_top_batch_ids_answer_types_by_features_outliers(
            self,
            features: list = ["grad_mean", "grad_abs_mean", "grad_median", "grad_std", "grad_dir", "grad_cos_dist", "var_train_loss"],
            outliers: list = [],
            epochs_by_class: dict[str, list] = {},
            rolling_window: int = 0,
            log_scale: bool = False,
            bins: int = 50,
            alpha: float = 0.7,
            color: str = 'blue',
        ):
            """
            Plot answer types by features outliers for batch IDs using histograms.
            """
            if len(list(epochs_by_class.keys())) == 0:
                return "No epochs found for the specified classes."
            
            fin_dataset = datasets.load_dataset("ibm-research/finqa", "en", split="train")
            answers = list(map(clean_answer, fin_dataset))
            answer_types = list(map(get_answer_formats, answers))
            all_unique_types, all_counts = np.unique(answer_types, return_counts=True)
                                
            outliers = outliers if len(outliers) > 0 else self.default_outliers

            if len(features) == 0:
                features = ["grad_mean", "grad_abs_mean", "grad_median", "grad_std", "grad_dir", "grad_cos_dist", "var_train_loss"]

            for feature in features:
                for class_name, epochs in epochs_by_class.items():
                    if len(epochs) == 0:
                        print(f"No epochs found for class {class_name}. Skipping.")
                        continue
                    
                    batch_ids = self.get_batch_ids_by_distribution_range(rolling_window, features=[feature], epoch_ids=epochs)[feature]
                    feature_cleaned = columns_cleaned.get(feature, feature)
                    title = f'{feature_cleaned} Outliers by Answer Types for Class {class_name}'
                    file_name = f'{feature_cleaned}_outliers_by_answer_types_for_class_{class_name}'
                    self._prepare_plot('Answer Types', 'Count (\%)', figsize=(width, height))
                    if log_scale:
                        plt.yscale('log')

                    for shift, ids in batch_ids.items():
                        if len(ids) > 0 and shift in outliers:
                            cls_batch_ids = ids
                            answer_types_for_batch = [answer_types[i] for i in cls_batch_ids]
                            unique_types, counts = np.unique(answer_types_for_batch, return_counts=True)
                            unique_types_idx = [np.where(all_unique_types == ut)[0][0] for ut in unique_types]
                            bar = counts / all_counts[unique_types_idx] * 100 if len(counts) > 0 else 0.
                            plt.hist(unique_types_idx, bins=bins, weights=bar, alpha=alpha, label=f'$\{class_name}$', color=color)

                    # plt.xticks(np.arange(len(all_unique_types)), all_unique_types)
                    plt.legend()
                    self._baseplot(title, plot=True, path="outliers")
            return self.files_to_save

    def plot_top_batch_ids_features_outliers_by_answer_types(
            self,
            features: list = ["grad_mean", "grad_abs_mean", "grad_median", "grad_std", "grad_dir", "grad_cos_dist", "var_train_loss"],
            outliers: list = [],
            epochs_by_class: dict[str, list] = {},
            rolling_window: int = 0,
            log_scale: bool = False,
            bins: int = 50,
            alpha: float = 0.7,
            color: str = 'blue',
        ):
            """
            Plot answer types by features outliers for batch IDs.
            """
            if len(list(epochs_by_class.keys())) == 0:
                return "No classes found for the specified epochs."
            
            fin_dataset = datasets.load_dataset("ibm-research/finqa", "en", split="train")
            answers = list(map(clean_answer, fin_dataset))
            answer_types = list(map(get_answer_formats, answers))
            all_unique_types, all_counts = np.unique(answer_types, return_counts=True)
                                
            outliers = outliers if len(outliers) > 0 else self.default_outliers
            colors = plt.get_cmap("viridis", len(epochs_by_class.keys()))
            colors = [colors(i) for i in range(len(epochs_by_class.keys()))]
            if len(features) == 0:
                features = ["grad_mean", "grad_abs_mean", "grad_median", "grad_std", "grad_dir", "grad_cos_dist", "var_train_loss"]

            for outlier in outliers:
                for feature in features:
                    feature_cleaned = columns_cleaned.get(feature, feature)
                    for answer_type in all_unique_types:
                        title = f'{feature_cleaned} Outliers ({outlier}) by Class for Answer Type {answer_type}'
                        _outlier = outlier.replace("$", "").replace("\\", "").replace(">", "sup").replace("<", "inf")
                        _answer_type = answer_type.replace(" ", "_").replace("'", "")
                        file_name = f'{feature}_outliers_{_outlier}_by_class_for_answer_type_{_answer_type}'
                        self._prepare_plot('Classes', 'Count (\%)', figsize=(width, height))
                        if log_scale:
                            plt.yscale('log')
                        b_width = 0
                        class_names = [ f"$\{class_name}$" for class_name in epochs_by_class.keys()]
                        bars = np.zeros(len(class_names))
                        for c_id, (class_name, epochs) in enumerate(epochs_by_class.items()):
                            if len(epochs) == 0:
                                print(f"No epochs found for class {class_name}. Skipping.")
                                continue
                            
                            batch_ids = self.get_batch_ids_by_distribution_range(rolling_window, features=[feature], epoch_ids=epochs)[feature]
                            for shift, ids in batch_ids.items():
                                if len(ids) > 0 and shift==outlier:
                                    cls_batch_ids = ids
                                    answer_types_for_batch = [answer_types[i] for i in cls_batch_ids]
                                    counts = sum(1 for at in answer_types_for_batch if at == answer_type)
                                    bar = counts / all_counts[np.where(all_unique_types == answer_type)[0][0]] * 100 if counts > 0 else 0.
                                    # plt.hist([bar], bins=bins, alpha=alpha, label=f'$\{class_name}$', color=colors(c_id))
                                    bars[c_id] += bar
                                    b_width += 1
                        if b_width == 0:
                            if self.verbose:
                                print(f"No outliers found for {feature} in the specified range.")
                            plt.close()
                            continue
                        plt.bar(class_names, bars, color=colors)

                        # plt.xticks(np.arange(b_width) + bar_width / 2, list(epochs_by_class.keys()))
                        # plt.legend()
                        self._baseplot(title, file_name=file_name, plot=True, path="outliers")
            return self.files_to_save
    
    def plot_average_confidence_by_cluster(
            self,
            ids_by_cluster: dict[str, list],
        ):
        """        Plot average confidence by cluster.
        """
        self.training_batch_data[["confidence_train", "epoch"]].groupby("epoch").mean().plot(
            y="confidence_train", 
            title="Average Confidence by Epoch",
            xlabel="Epoch",
            ylabel="Average Confidence",
            figsize=(width, height)
        )
        plt.show()
        for cluster, ids in ids_by_cluster.items():
            cluster_data = self.training_batch_data[self.training_batch_data['batch_ids'].apply(lambda x: any(int(i) in ids for i in x.strip('[]').split(',')))]
            if len(cluster_data) == 0:
                print(f"No data found for cluster \\{cluster}. Skipping.")
                continue
            grouped_data = cluster_data[["confidence_train", "epoch"]].groupby("epoch")
            mean_data = grouped_data.mean()
            std_data = grouped_data.std()
            mean_data.plot(
                y="confidence_train", 
                yerr=std_data["confidence_train"],
                title=f"Average Confidence for Cluster \\{cluster} by Epoch (with Std Dev)",
                xlabel="Epoch",
                ylabel="Average Confidence",
                figsize=(width, height),
                capsize=4
            )
            plt.show()
            self._baseplot(f"Average Confidence for Cluster \\{cluster} by Epoch (with Std Dev)", plot=True)

    def plot_all(self):
        """
        Plot all available data.
        """
        self.plot_grad_data(show_outliers=False)
        self.plot_epoch_data()
        self.plot_couple_data()
        self.plot_grad_histograms()
        # self.get_seasonality(data_type="batch")
        # self.get_seasonality(data_type="epoch")
        # self.plot_batch_ids_by_distribution_range()
        # self.plot_answer_types(log_scale=False, rolling_window=0)
        self.plot_excepted_calibration_error()
        self.plot_answer_types(log_scale=True, rolling_window=0)
        # self.plot_answer_types(log_scale=False, rolling_window=10)
