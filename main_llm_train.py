from llm.pipeline_train import train_llm_pipeline

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import argparse
from dotenv import load_dotenv

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
    parser.add_argument('--train-mode', type=str, default="batch", metavar='TM',)
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
    load_dotenv()
    # from utils import get_device
    # torch.cuda.set_per_process_memory_fraction(1., device=get_device())
    # torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    from random import randint
    WORLD_SIZE = 1
    master_port = f'{randint(10_000,40_000)}'
    if WORLD_SIZE == 0:
        print("No GPU available")
    if WORLD_SIZE<=1:
        train_llm_pipeline(rank=0, world_size=1, master_port=master_port, args=args)
    else:
        print("WORLD SIZE = ", WORLD_SIZE)
        try:
            print("try mp spawn")

            torch.multiprocessing.set_start_method("fork", force=True)
            mp.spawn(train_llm_pipeline,
                args=(WORLD_SIZE, master_port, args),
                nprocs=WORLD_SIZE,
                join=True
            )
        
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()