from archives.main_llm import main_train

from llm.data.dataset import get_data
from llm.data.tokenizer import Tokenizer, CONTROL_TOKENS, CONTROL_TOKENS_LIST

from llm.model import LLM
from llm.parallel_trainer import ParallelTrainer
from llm.eval import EvalTask

from tm_data.preprocessing import InputCSV

from utils import get_device, get_cuda_allocation

import os
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    torch.cuda.empty_cache()

def cleanup():
    dist.destroy_process_group()

def print_fsdp_wrapping(module, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, FSDP):
            print(f"{full_name} is wrapped with FSDP")
        else:
            print(f"{full_name} is not wrapped with FSDP")
        print_fsdp_wrapping(child, full_name)

def fsdp_main(rank, world_size, args):
    print("rank", rank)
    setup(rank, world_size)

    try:
        tokenizer = Tokenizer(model_name=args.tokenizer)
    except:
        tokenizer = Tokenizer()

    tokenizer.add_special_tokens(CONTROL_TOKENS_LIST)

    # TODO: add to settings
    max_length = 1024
    max_q_length = 1024
    max_a_length = 8
    short_answer = True
    dataset_params = {
        "tokenizer": tokenizer,
        "max_length": max_length,
        "max_q_length": max_q_length,
        "max_a_length": max_a_length,
        "short_answer": short_answer
    }
    train, test, val = get_data(**dataset_params)
    
    sampler1 = DistributedSampler(train, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val, rank=rank, num_replicas=world_size, shuffle=True)
    sampler3 = DistributedSampler(test, rank=rank, num_replicas=world_size)

    train_kwargs = { 'batch_size': args.batch_size, 'sampler': sampler1 }
    test_kwargs = { 'batch_size': args.test_batch_size, 'sampler': sampler2 }
    cuda_kwargs = {
        'num_workers': 2,
        'pin_memory': True,
        # 'shuffle': True
    }
    
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    # torch.cuda.set_device(rank)

    csv_path = "dataset/uq_features"
    
    vocab_size = tokenizer.get_vocab_size()
    max_content = max(train.max_q_len, val.max_a_len)

    model_hyperparams = { 
        "vocab_size": vocab_size, 
        "model_size": 512,
        "max_content": max_content, 
        "nhead": 8, 
        "num_encoder_layers": 4, 
        "num_decoder_layers": 4
    }
    model = LLM(**model_hyperparams).to(rank)

    if args.verbose:
        model.summary()

    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN/Inf detected in parameter {name}")
            
    model_mem_required = model.memory_storage()
    model = FSDP(model,
        use_orig_params=world_size > 1,
        # auto_wrap_policy=my_auto_wrap_policy,
        # cpu_offload=CPUOffload(offload_params=True)
    )
    # model.embedding = FSDP(model.embedding, use_orig_params=True) 

    if args.verbose:
        print_fsdp_wrapping(model)
        print(f"Model memory size: {model_mem_required:,} bytes")
        print("FSDP model:", model)
    
    opt = optim.AdamW(model.parameters(), lr=args.lr / args.batch_size)
    lr_scheduler = StepLR(opt, step_size=80, gamma=args.gamma)

    loss_mask = torch.ones(vocab_size)

    for index in [tokenizer.soa_token_id, tokenizer.eoa_token_id, tokenizer.pad_token_id]:
        loss_mask[index] = 0

    loss_mask = loss_mask.float().to(rank)
    criterion = nn.CrossEntropyLoss(
        weight=loss_mask, 
        ignore_index=tokenizer.special_tokens[CONTROL_TOKENS.padding],
        reduction="sum"
    )
    # scheduler = StepLR(opt, step_size=1, gamma=args.gamma)
    # init_start_event.record()
    
    torch.cuda.set_per_process_memory_fraction(0.9, device=rank)
    torch.backends.cudnn.benchmark = True
    free_memory = get_cuda_allocation(verbose=args.verbose)

    if args.verbose and False:
        assert free_memory >= model_mem_required, f"""MEMORY ERROR: Free memory space is to small to store the model on cuda. Try to allocate more memory. 
            Current rank: {rank}.
            Free memory: {free_memory:,} bytes.
            Memory required for the model: {model_mem_required:,} bytes"""
    
    eval_task = EvalTask(tokenizer=tokenizer)
    csv_object = InputCSV(
        model=model, 
        path=csv_path,
        world_size=world_size,
        eval_metrics=eval_task.result_keys
    )
    trainer = ParallelTrainer(
        model,
        optimizer=opt,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        csv_object=csv_object,
        rank=rank,
        world_size=world_size,
        eval_task=eval_task,
        name="make-tm-dataset.test",
        soa_token_id=tokenizer.soa_token_id,
        eoa_token_id=tokenizer.eoa_token_id,
        pad_token_id=tokenizer.pad_token_id,
        len_answer=val.max_a_len
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
            train_kwargs=train_kwargs,
            val_kwargs=test_kwargs,
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
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False, 
                        help='For Saving the current Model')
    parser.add_argument('--verbose', action='store_true', default=False, 
                        help='For Showing some more information')
    parser.add_argument('--patience', type=int, default=25, metavar='P',
                        help='Early stopping patience (default: 25)')
    parser.add_argument('--min-delta', type=float, default=0.05, metavar='D',
                        help='Minimum delta for early stopping (default: 0.05)')
    parser.add_argument(
        "-st",
        "--skip_training",
        help="Skip the training and only evaluate the model",
        action="store_true",
    )
    parser.add_argument('--model-dir', type=str, default="models/", metavar='MD',
                        help='Directory where the model is saved')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    print(f"WORLD_SIZE: {WORLD_SIZE}")

    if WORLD_SIZE == 0:
        print("No GPU available")
        main_train(args)
    elif WORLD_SIZE==1:
        fsdp_main(rank=0, world_size=1, args=args)
    else:
        mp.spawn(fsdp_main,
            args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,
            join=True
        )