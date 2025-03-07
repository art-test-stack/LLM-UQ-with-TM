from archives.main_llm import main_train
from pipeline_train_llm import train_llm_pipeline

import torch
import torch.multiprocessing as mp

import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Train a LLM on Financial Risk Management Q&A dataset.",
        epilog="Enjoy the program! :)",
    )
    parser.add_argument('--params_file', type=str, default="params.yaml", metavar='PF',
                        help='File containing the model parameters (default: params.yaml)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False, 
                        help='For Saving the current Model')
    parser.add_argument('--verbose', action='store_true', default=False, 
                        help='For Showing some more information')
    parser.add_argument(
        "-st",
        "--skip_training",
        help="Skip the training and only evaluate the model",
        action="store_true",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    print(f"WORLD_SIZE: {WORLD_SIZE}")

    if WORLD_SIZE == 0:
        print("No GPU available")
        main_train(args)
    elif WORLD_SIZE==1:
        train_llm_pipeline(rank=0, world_size=1, args=args)
    else:
        mp.spawn(train_llm_pipeline,
            args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,
            join=True
        )