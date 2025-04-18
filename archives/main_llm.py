from llm.data.dataset import get_data
from llm.data.tokenizer import Tokenizer

from llm.model import LLM
from archives.trainer import Trainer

from tm_data.fetch_data import InputCSV

from utils import get_device, get_cuda_allocation

import pandas as pd
import datasets

from torch import optim, nn

import argparse

def main_train(args):
    print("main train function")
    device = get_device()
    tokenizer = Tokenizer()
    csv_path = "dataset/uq_features"
    
    train, test, val = get_data(tokenizer)

    vocab_size = tokenizer.get_vocab_size()
    max_content = max(train.max_content, val.max_content)
    model_hyperparams = { 
        "vocab_size": vocab_size, 
        "model_size": 32,
        "max_content": max_content, 
        "nhead": 1, 
        "num_encoder_layers": 1, 
        "num_decoder_layers": 1
    }
    model = LLM(**model_hyperparams)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.summary()
    # import sys
    # print("model size:", print(sys.getsizeof(train)))
    print(f"mem storage: {model.memory_storage():,} bytes")
    csv_object = InputCSV(model, csv_path)
    get_cuda_allocation()

    trainer = Trainer(
        model,
        optimizer=opt,
        criterion=criterion,
        csv_object=csv_object,
        device=device
    )
    try:
        trainer.load_model()
    except:
        print("No model found")

    if not args.skip_training:
        trainer.fit(
            train, 
            val, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            min_delta=args.min_delta,
        )

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Train a Tsetlin Machine to evaluate the uncertainty of a LLM",
        epilog="Enjoy the program! :)",
    )
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--patience', type=int, default=5, metavar='P',
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--min-delta', type=float, default=0.05, metavar='D',
                        help='Minimum delta for early stopping (default: 0.05)')
    parser.add_argument(
        "-st",
        "--skip_training",
        help="Skip the training and only evaluate the model",
        action="store_true",
    )
    args = parser.parse_args()

    main_train(args)
    
 
    # import torch
    # import matplotlib.pyplot as plt
    # import numpy as np

    # grads = trainer.model.get_grads()
    # spectra = []
    # for grad in grads:
    #     if grad.ndim == 2:
    #         u, s, v = torch.svd(grad)
    #         spectra.append(s)
    # print("Spectra of all matrices in grads:")
    
    # for i, spectrum in enumerate(spectra):
    #     mean_spectrum = torch.mean(spectrum).item()
    #     median_spectrum = torch.median(spectrum).item()
    #     std_spectrum = torch.std(spectrum).item()
    #     max_spectrum = torch.max(spectrum).item()
    #     min_spectrum = torch.min(spectrum).item()

    #     print(f"\nMatrix {i+1} Spectrum Stats:")
    #     print(f"Mean: {mean_spectrum}")
    #     print(f"Median: {median_spectrum}")
    #     print(f"Standard Deviation: {std_spectrum}")
    #     print(f"Maximum: {max_spectrum}")
    #     print(f"Minimum: {min_spectrum}")

    # print('\nAll Spectra')
    # all_spectras = torch.cat([spectrum.view(-1) for spectrum in spectra])
    # mean_all_spectras = torch.mean(all_spectras).item()
    # median_all_spectras = torch.median(all_spectras).item()
    # std_all_spectras = torch.std(all_spectras).item()
    # max_all_spectras = torch.max(all_spectras).item()
    # min_all_spectras = torch.min(all_spectras).item()

    # print(f"Mean of all spectras: {mean_all_spectras}")
    # print(f"Median of all spectras: {median_all_spectras}")
    # print(f"Standard deviation of all spectras: {std_all_spectras}")
    # print(f"Maximum spectrum value: {max_all_spectras}")
    # print(f"Minimum spectrum value: {min_all_spectras}")
    # all_grads = torch.cat([grad.view(-1) for grad in grads])

    # print("\nall_grads", all_grads.shape)

    # mean_grad = torch.mean(all_grads).item()
    # median_grad = torch.median(all_grads).item()
    # std_grad = torch.std(all_grads).item()
    # max_grad = torch.max(all_grads).item()
    # min_grad = torch.min(all_grads).item()

    # print(f"Mean of gradients: {mean_grad}")
    # print(f"Median of gradients: {median_grad}")
    # print(f"Standard deviation of gradients: {std_grad}")
    # print(f"Maximum gradient value: {max_grad}")
    # print(f"Minimum gradient value: {min_grad}")

    # quantiles = [0.1,0.15,0.20,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.70,0.75,0.80,0.85,0.90]
    # # quantiles = [1e-2,.05,0.1,0.15,0.20,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1-1e-2]
    
    # all_grads_np = all_grads.cpu().detach().numpy()
    # grad_bins = np.quantile(all_grads_np, quantiles).tolist()
    
    # # print(f"0 is {"" if 0 in all_grads_np else "not "}in all_grads_np")
    # # all_grads_np_log = np.log(all_grads_np)
    # # grad_bins_log = np.quantile(all_grads_np_log, quantiles).tolist()

    # all_spectras_np = all_spectras.cpu().detach().numpy()
    # alpha, beta = compute_beta_dist_params(mean_all_spectras, std_all_spectras)
    # print("\nalpha, beta", alpha, beta)
    # sp_bins = np.quantile(all_spectras_np, quantiles).tolist()

    # all_spectras_np_log = np.log(all_spectras_np)
    # # alpha_log, beta_log = compute_beta_dist_params(np.mean(all_spectras_np_log), np.std(all_spectras_np_log))
    # # print("\nalpha_log, beta_log", alpha_log, beta_log)
    # sp_bins_log = np.quantile(all_spectras_np_log, quantiles).tolist()
    
    # print("\nSpectra Bins", sp_bins)
    # print("\nGrad Bins", grad_bins)

    
    # from scipy.stats import beta as beta_dist

    # plt.hist(all_spectras_np_log, bins=sp_bins_log, alpha=0.5, edgecolor='black', label='spectra')
    # plt.plot(sp_bins_log, beta_dist.logpdf(sp_bins_log, alpha, beta), 'r-', lw=5, alpha=0.6, label='beta pdf')
    # plt.title("Distribution of Spectras Values")
    # plt.xlabel("Spectra Value")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.show()

    # # plt.hist(all_grads_np_log, bins=grad_bins_log, alpha=0.5, edgecolor='black', log=True)
    # # plt.title("Distribution of Gradient Values")
    # # plt.xlabel("Gradient Value")
    # # plt.ylabel("Frequency")
    # # plt.show()

    # plt.hist(all_spectras_np, bins=sp_bins, alpha=0.5, edgecolor='black', label='spectra')
    # plt.plot(sp_bins, beta_dist.pdf(sp_bins, alpha, beta), 'r-', lw=5, alpha=0.6, label='beta pdf')
    # plt.title("Distribution of Spectras Values")
    # plt.xlabel("Spectra Value")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.show()

    # plt.hist(all_grads_np, bins=grad_bins, alpha=0.5, edgecolor='black', log=True)
    # plt.title("Distribution of Gradient Values")
    # plt.xlabel("Gradient Value")
    # plt.ylabel("Frequency")
    # plt.show()


    # last_grads = get_last_layers_grads(grads)
    # last_spectrums = get_last_layers_grads(spectra)

    # last_grads_np = torch.cat([grad.view(-1) for grad in last_grads]).cpu().detach().numpy()
    # last_spectrums_np = torch.cat([spectrum.view(-1) for spectrum in last_spectrums]).cpu().detach().numpy()

    # last_grads_bins = np.quantile(last_grads_np, quantiles).tolist()
    # last_spectrums_bins = np.quantile(last_spectrums_np, quantiles).tolist()

    # alpha, beta = compute_beta_dist_params(np.mean(last_spectrums_np), np.std(last_spectrums_np))
    # print("\nLast layers alpha, beta:", alpha, beta)
    
    # plt.hist(last_spectrums_np, bins=last_spectrums_bins, alpha=0.5, edgecolor='black', label='spectra')
    # plt.plot(last_spectrums_bins, beta_dist.pdf(last_spectrums_bins, alpha, beta), 'r-', lw=5, alpha=0.6, label='beta pdf')
    # plt.title("Distribution of Spectras Values")
    # plt.xlabel("Spectra Value")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.show()

    # plt.hist(last_grads_np, bins=last_grads_bins, alpha=0.5, edgecolor='black', log=True)
    # plt.title("Distribution of Gradient Values")
    # plt.xlabel("Gradient Value")
    # plt.ylabel("Frequency")
    # plt.show()

    # print("\nLast Layers Spectra")
    # for i, spectrum in enumerate(last_spectrums):
    #     mean_spectrum = torch.mean(spectrum).item()
    #     median_spectrum = torch.median(spectrum).item()
    #     std_spectrum = torch.std(spectrum).item()
    #     max_spectrum = torch.max(spectrum).item()
    #     min_spectrum = torch.min(spectrum).item()

    #     print(f"\nMatrix {i+1} Spectrum Stats:")
    #     print(f"Mean: {mean_spectrum}")
    #     print(f"Median: {median_spectrum}")
    #     print(f"Standard Deviation: {std_spectrum}")
    #     print(f"Maximum: {max_spectrum}")
    #     print(f"Minimum: {min_spectrum}")

    # print("\nLast Layers Gradients")
    # for i, grad in enumerate(last_grads):
    #     mean_grad = torch.mean(grad).item()
    #     median_grad = torch.median(grad).item()
    #     std_grad = torch.std(grad).item()
    #     max_grad = torch.max(grad).item()
    #     min_grad = torch.min(grad).item()

    #     print(f"\nMatrix {i+1} Gradient Stats:")
    #     print(f"Mean: {mean_grad}")
    #     print(f"Median: {median_grad}")
    #     print(f"Standard Deviation: {std_grad}")
    #     print(f"Maximum: {max_grad}")
    #     print(f"Minimum: {min_grad}")

    