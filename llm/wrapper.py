import torch
from torch import nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import functools

def fsdp_wrapper(
        model: nn.Module,
        model_block_cls: nn.Module,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
        device_id: int = 0,
    ):
    if not torch.cuda.is_available():
        return model
    print("Load FSDP wrapper...")
    device_id = torch.cuda.current_device()
    transformer_auto_wrapper_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            model_block_cls,
        },
    )
    model.layers = FSDP(
        model.layers,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=transformer_auto_wrapper_policy,
        device_id=device_id,
    )
    print("FSDP model", model)
    print("FSDP wrapper loaded!")
    return model
    