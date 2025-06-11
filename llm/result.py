from llm.utils import get_model_dir
from llm.data.dataset import clean_answer, get_answer_formats

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datasets

class TrainingResult:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_dir = get_model_dir(model_name, return_error=True)
        self.training_batch_data = pd.read_csv(self.model_dir.joinpath("fetched_batch_data.csv")).iloc[1:,]
        self.training_epoch_data = pd.read_csv(self.model_dir.joinpath("fetched_training_data.csv")).iloc[1:,]

        self.default_outliers = ["<-4σ", ">4σ"]

    def __repr__(self):
        return f"TrainingResult(model_name={self.model_name}, training_time={self.training_time}, loss={self.loss})"
    
    def plot_grad_data(self, show_outliers: bool = False):
        
        for col in self.training_batch_data.columns:
            if not col == "batch_ids":
            # if col.startswith('grad_'):
            #     plt.plot(self.training_batch_data[col], label=col)
                plt.figure(figsize=(10, 5))
                plt.plot(self.training_batch_data[col])
                if show_outliers:
                    outliers = self.training_batch_data[col].astype(float).abs() > 3 * self.training_batch_data[col].astype(float).std()
                    plt.scatter(np.where(outliers)[0], self.training_batch_data[col][outliers], color='red', label='Outliers')
                plt.xlabel('Step')
                plt.ylabel(f'{col} Value')
                plt.title(f'{col} for {self.model_name}')
                plt.show()

    def plot_epoch_data(self):
        for col in self.training_epoch_data.columns:
            
            if not col == "epoch":
                plt.figure(figsize=(10, 5))
                plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data[col], label=col)
                plt.xlabel('Epoch')
                plt.ylabel(f'{col} Value')
                plt.title(f'{col} for {self.model_name}')
                plt.legend()
                plt.show()
    
    def plot_couple_data(self):
        m_list = ["accuracy", "recall", "precision", "f1", "confidence", "perplexity"]
        for metric in m_list:
            plt.figure(figsize=(10, 5))
            plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data[f'{metric}_train'], label=f'{metric}_train')
            plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data[f'{metric}_val'], label=f'{metric}_val')
            plt.xlabel('Epoch')
            plt.ylabel(f'{metric} Value')
            plt.title(f'{metric} for {self.model_name}')
            plt.legend()
            plt.show()

        
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data['train_loss'], label='train_loss')
        plt.plot(self.training_epoch_data['epoch'], self.training_epoch_data['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Losses Value')
        plt.title(f'Losses for {self.model_name}')
        plt.legend()
        plt.show()
    
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
            plt.figure(figsize=(12, 5))
            plt.plot(t, data, label=f'Original {metric}')
            plt.plot(t, reconstructed, label=f'Reconstructed {metric} (top {top_seasonalities} seasonalities)')
            plt.xlabel('Step')
            plt.ylabel(metric)
            plt.title(f'Seasonality in {metric} for {self.model_name}')
            plt.legend()
            plt.show()


    def get_batch_ids_by_distribution_range(self, rolling_window: int = 0):
        """
        Get batch ids by distribution range.
        """
        features = ["grad_abs_mean", "grad_mean", "grad_median", "grad_std", "grad_dir", "grad_cos_dist", "var_train_loss"]
        if rolling_window > 0:
            # Apply rolling mean and std if rolling_window is specified
            mean = self.training_batch_data[features].astype(float).rolling(window=rolling_window, min_periods=1).mean().iloc[rolling_window:]
            std = self.training_batch_data[features].astype(float).rolling(window=rolling_window, min_periods=1).std().iloc[rolling_window:]
        else:
            mean = self.training_batch_data[features].astype(float).mean(axis=0)
            std = self.training_batch_data[features].astype(float).std(axis=0)

        distribution_ranges = {
            "<-4σ": lambda x, m, s: x < m - 4 * s,
            "-4σ--3σ": lambda x, m, s: (x >= m - 4 * s) & (x < m - 3 * s),
            "-3σ--2σ": lambda x, m, s: (x >= m - 3 * s) & (x < m - 2 * s),
            "-2σ--1σ": lambda x, m, s: (x >= m - 2 * s) & (x < m - 1 * s),
            "-1σ-1σ": lambda x, m, s: (x >= m - 1 * s) & (x <= m + 1 * s),
            "1σ-2σ": lambda x, m, s: (x > m + 1 * s) & (x <= m + 2 * s),
            "2σ-3σ": lambda x, m, s: (x > m + 2 * s) & (x <= m + 3 * s),
            "3σ-4σ": lambda x, m, s: (x > m + 3 * s) & (x <= m + 4 * s),
            ">4σ": lambda x, m, s: x > m + 4 * s,
        }
        # distribution_ranges = {
        #     "<-3σ": lambda x, m, s: x < m - 3 * s,
        #     "-3σ--2σ": lambda x, m, s: (x >= m - 3 * s) & (x < m - 2 * s),
        #     "-2σ--1σ": lambda x, m, s: (x  >= m - 2 * s) & (x < m - 1 * s),
        #     "-1σ-1σ": lambda x, m, s: (x >= m - 1 * s) & (x <= m + 1 * s),
        #     "1σ-2σ": lambda x, m, s: (x > m + 1 * s) & (x <= m + 2 * s),
        #     "2σ-3σ": lambda x, m, s: (x > m + 2 * s) & (x <= m + 3 * s),
        #     ">3σ": lambda x, m, s: x > m + 3 * s,
        # }
        batch_ids = { feature: {shift: [] for shift in distribution_ranges} for feature in features }
        data = self.training_batch_data[features].astype(float).iloc[rolling_window:]
        # Convert string representations of lists into actual lists of int
        batch_ids_arr = self.training_batch_data['batch_ids'].apply(lambda x: list(map(int, x.strip('[]').split(',')))).values
        for feature in features:
            for shift, cond in distribution_ranges.items():
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
            plt.figure(figsize=(12, 6))
            for shift, ids in shifts.items():
                if len(ids) > 0 and shift in outliers:
                    plt.bar(ids.keys(), ids.values(), label=shift)
            
            if log_scale:
                plt.yscale('log')
            plt.xlabel('Batch IDs')
            plt.ylabel('Count')
            plt.title(f'Batch IDs by Distribution Range for {feature}')
            plt.legend()
            plt.show()

    
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
            plt.figure(figsize=(10, 6))
            if log_scale:
                plt.yscale('log')
            x = np.arange(len(all_unique_types))
            width = 0
            for c, idx in shifts.items():
                if len(idx) > 0 and c in outliers:
                    cls_batch_ids = idx
                    answer_types_for_batch = [answer_types[i] for i in cls_batch_ids]
                    unique_types, counts = np.unique(answer_types_for_batch, return_counts=True)
                    unique_types_idx = [np.where(all_unique_types == ut)[0][0] for ut in unique_types]
                    bar = counts / all_counts[unique_types_idx] * 100 if len(counts) > 0 else 0.
                    plt.bar(np.array(unique_types_idx) + bar_width * width, bar, width=bar_width, label=f'Range {c}', alpha=0.7)
                    width += 1
            # plt.bar(x, all_counts, width=bar_width, label='All Answer Types', alpha=0.7)
            if width == 0:
                print(f"No outliers found for {feature} in the specified range.")
                continue
            plt.xticks(x + bar_width * (width - 1) / 2 , all_unique_types)
            plt.xlabel("Answer Types")
            plt.ylabel("Count (%)")
            plt.legend()
            # if self.plot_titles:
            plt.title(f"Answer Type Distribution with respect to {feature} outliers")
            # else:
            #     print(f"Answer Type Distribution for Cluster {c} - Run {self.run}")
            plt.show()

    def plot_answer_types_outliers(self, log_scale: bool = False, rolling_window: int = 0, outliers: list = []):
        """
        Plot answer types for outliers.
        """
        outliers = outliers if len(outliers) > 0 else self.default_outliers
        batch_ids = self.get_batch_ids_by_distribution_range(rolling_window)
        for feature, shifts in batch_ids.items():
            plt.figure(figsize=(20, 6))
            if log_scale:
                plt.yscale('log')
            for shift, ids in shifts.items() and shift in outliers:
                if len(ids) > 0:
                    plt.bar(ids.keys(), ids.values(), label=shift)
            plt.xlabel('Batch IDs')
            plt.ylabel('Count')
            plt.title(f'Batch IDs by Distribution Range for {feature}')
            plt.legend()
            plt.show()

    def plot_excepted_calibration_error(self, plot: bool = True):
        fin_dataset = datasets.load_dataset("ibm-research/finqa", "en", split="train")
        n = len(fin_dataset)
        data = self.training_batch_data[["accuracy_train","confidence_train","epoch"]]
        first_bin = self.training_batch_data['batch_ids'].apply(lambda x: list(map(int, x.strip('[]').split(',')))).values[0]
        bin_size = len(first_bin)

        ece_scores = data.groupby('epoch').apply(
            lambda x: np.abs(x['accuracy_train'], x['confidence_train']) * bin_size / n
        )
        if plot:
            plt.figure(figsize=(10, 5))
        plt.plot(data["epoch"], ece_scores, label=self.model_name, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Expected Calibration Error')
        plt.title(f'Expected Calibration Error for {self.model_name}')
        plt.grid()
        if plot:
            plt.show()
        else:
            return ece_scores

    def plot_all(self):
        """
        Plot all available data.
        """
        self.plot_grad_data()
        self.plot_epoch_data()
        self.plot_couple_data()
        # self.get_seasonality(data_type="batch")
        # self.get_seasonality(data_type="epoch")
        # self.plot_batch_ids_by_distribution_range()
        # self.plot_answer_types(log_scale=False, rolling_window=0)
        self.plot_excepted_calibration_error()
        self.plot_answer_types(log_scale=True, rolling_window=0)
        self.plot_answer_types(log_scale=False, rolling_window=10)