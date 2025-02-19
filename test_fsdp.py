from main_llm import main_train

from data.dataset import get_data
from data.tokenizer import Tokenizer

from llm.model import LLM
from llm.parallel_trainer import ParallelTrainer, setup, cleanup

from tm_data.preprocessing import InputCSV

from utils import get_device

import pandas as pd
import datasets

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import argparse

import functools


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    tokenizer = Tokenizer()
    train, test, val = get_data(tokenizer)
    
    sampler1 = DistributedSampler(train, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val, rank=rank, num_replicas=world_size)
    sampler3 = DistributedSampler(test, rank=rank, num_replicas=world_size)

    train_kwargs = { 'batch_size': args.batch_size, 'sampler': sampler1 }
    test_kwargs = { 'batch_size': args.test_batch_size, 'sampler': sampler2 }
    cuda_kwargs = {
        'num_workers': 2,
        'pin_memory': True,
        'shuffle': False
    }
    
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val, **test_kwargs)

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)

    csv_path = "dataset/uq_features"
    
    vocab_size = tokenizer.get_vocab_size()
    max_content = max(train.max_content, val.max_content)

    model = LLM(vocab_size=vocab_size, max_content=max_content).to(rank)
    # model = FSDP(model)
    model = FSDP(model,
        auto_wrap_policy=my_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True))

    print("FSDP model:", model)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    scheduler = StepLR(opt, step_size=1, gamma=args.gamma)
    # init_start_event.record()
    
    csv_object = InputCSV(model, csv_path)
    trainer = ParallelTrainer(
        model,
        optimizer=opt,
        criterion=criterion,
        csv_object=csv_object,
        rank=rank
    )
    # Model checkpoint saving, by saving to the rank0 CPU
    # https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html#model-checkpoint-saving-by-streaming-to-the-rank0-cpu

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
            train_sampler=sampler1
        )

    if rank == 0:
        pass
        # init_end_event.synchronize()
        # print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        # print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            pass
            # torch.save(states, "mnist_cnn.pt")

    cleanup()


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

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    print("WORLD_SIZE:", WORLD_SIZE)

    if WORLD_SIZE == 0:
        print("No GPU available")
        main_train(args)
    else:
        mp.spawn(fsdp_main,
            args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,
            join=True
        )