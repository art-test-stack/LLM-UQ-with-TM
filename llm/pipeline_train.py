from llm.data.dataset import get_data
from llm.handlers.handler import model_handler
from llm.trainer import Trainer
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
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from llama_models.llama3.reference_impl.generation import Llama
import argparse
import functools
import yaml


def setup(rank, world_size, master_port: str = '12355'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    torch.cuda.empty_cache()
    print(f"Process {rank} initialized!")

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

def train_llm_pipeline(rank, world_size, args):
    print("rank", rank)
    setup(rank, world_size)

    with open(args.params_file, "r") as file:
        params = yaml.safe_load(file)

    model_params = params["model"]
    training_params = params["training"]
    data_params = params["data"]

    print("Load model and tokenizer...")
    model, tokenizer, TransformerBlock = model_handler(model_params)
    print("Model and tokenizer loaded!")
    model = model.to(rank)

    # TODO: add to settings
    dataset_params = {
        "tokenizer": tokenizer,
        "max_length": model_params["config"]["max_seq_len"],
        "max_a_length": 8, # TODO: TEMPORARY
        **data_params
    }
    start_pos = dataset_params["max_length"] - dataset_params["max_a_length"]

    print("Load FinQADataset...")
    train, test, val = get_data(**dataset_params)
    
    sampler1 = DistributedSampler(train, rank=rank, num_replicas=world_size, shuffle=False)
    sampler2 = DistributedSampler(val, rank=rank, num_replicas=world_size, shuffle=False)
    sampler3 = DistributedSampler(test, rank=rank, num_replicas=world_size)

    train_kwargs = { 'batch_size': training_params["batch_size"], 'sampler': sampler1 }
    test_kwargs = { 'batch_size': training_params["test_batch_size"], 'sampler': sampler2 }
    cuda_kwargs = {
        'num_workers': 0,
        'pin_memory': True,
        # 'shuffle': True
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    print("FinQADataset loaded!")
    
    vocab_size = tokenizer.get_vocab_size()

    if args.verbose:
        try:
            model.summary()
        except:
            print("No model summary available")

    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN/Inf detected in parameter {name}")
    
    if args.verbose:
        try:
            model_mem_required = model.memory_storage()
            # print_fsdp_wrapping(model)
            print(f"Model memory size: {model_mem_required:,} bytes")
        except:
            print("No model memory storage available")
        
        print("Model layers:", model)

    for i in model.named_parameters():
        print(f"{i[0]} -> {i[1].device}")

    print("FSDP Layers printed")
    torch.cuda.set_device(rank)
    print("Load FSDP wrapper...")
    transformer_auto_wrapper_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            TransformerBlock,
        },
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=transformer_auto_wrapper_policy,
        device_id=torch.cuda.current_device(),
    )
    print("FSDP model", model)
    print("Memory summary", torch.cuda.memory_summary(device=rank, abbreviated=False))
    print("FSDP wrapper loaded!")
    lr = float(training_params["learning_rate"]) / training_params["batch_size"]
    optim_params = {
        "lr": lr,
        "weight_decay": training_params["weight_decay"],
        "betas": training_params["betas"],
    }
    if training_params["optimizer"] == "adam":
        opt = optim.Adam(model.parameters(), **optim_params)
    elif training_params["optimizer"] == "adamw":
        opt = optim.AdamW(model.parameters(), **optim_params)
    else:
        raise ValueError("Invalid optimizer")
    
    print("Optimizer initialized")
    lr_scheduler = StepLR(opt, step_size=training_params["step_size"], gamma=training_params["gamma"])

    # loss_mask = torch.ones(vocab_size).to(rank)
    
    # for index in [tokenizer.pad_token_id]:
    #     loss_mask[index] = 0

    loss_mask = loss_mask.float().to(rank)
    criterion = nn.CrossEntropyLoss(
        # weight=loss_mask, 
        ignore_index=tokenizer.pad_token_id,
        reduction="sum"
    )
    
    free_memory = get_cuda_allocation(verbose=args.verbose)

    if args.verbose and False:
        assert free_memory >= model_mem_required, f"""MEMORY ERROR: Free memory space is to small to store the model on cuda. Try to allocate more memory. 
            Current rank: {rank}.
            Free memory: {free_memory:,} bytes.
            Memory required for the model: {model_mem_required:,} bytes"""
    
    eval_task = EvalTask(tokenizer=tokenizer)

    csv_path = os.getenv(data_params["uq_path"]) + f"_{model_params['name']}"
    print("CSV path:", csv_path)
    csv_object = InputCSV(
        model=model, 
        path=csv_path,
        world_size=world_size,
        eval_metrics=eval_task.result_keys
    )
    trainer = Trainer(
        model,
        optimizer=opt,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        csv_object=csv_object,
        rank=rank,
        world_size=world_size,
        eval_task=eval_task,
        name=model_params["name"],
        model_dir=os.getenv(model_params["dir"]),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        start_pos=start_pos
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
            train_kwargs=train_kwargs,
            val_kwargs=test_kwargs,
            **training_params
        )

    if rank == 0:
        pass

    cleanup()