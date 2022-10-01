# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

# from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import einops
import tqdm

import random
import time

# from google.colab import drive
from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "colab"
import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc
import collections
import copy

# import comet_ml
import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import transformers
from datasets import load_dataset
import json
from transformers import AutoTokenizer
import transformers
import datasets
import time
import wandb
import accelerate

from accelerate import Accelerator
from accelerate.utils import set_seed, write_basic_config
from accelerate import notebook_launcher
import os

from pprint import pprint

from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any, Optional

from easy_transformer.EasyTransformerConfig import EasyTransformerConfig
# %%
from IPython import get_ipython
ipython = get_ipython()
# Code to automatically update the EasyTransformer code as its edited without restarting the kernel
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")
import plotly.io as pio
pio.renderers.default = "vscode"
  

# %%
@dataclass
class TrainingConfig(EasyTransformerConfig):
    lr=1e-4
    d_model = None
    d_head = None
    n_heads = None
    n_ctx = None
    n_layers = None
    lurp = 100

cfg = TrainingConfig(
    d_model=128,
    n_layers = 2,
    n_heads = 4,
    d_head = 32,
    n_ctx=1024,
    d_mlp=512,
    act_fn='gelu',
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    # lr=100 
    )
print(cfg)
# cfg = EasyTransformerConfig(
#     d_model=128,
#     n_layers = 2,
#     n_heads = 4,
#     d_head = 32,
#     n_ctx=1024,
#     lr=100,
#     act_fn='gelu'
#     )
# print(cfg)
# %%
from easy_transformer import EasyTransformer
import transformers
# tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = EasyTransformer.from_config(cfg)
print(model)

# %%

def create_cfg(accelerator):
    cfg = {
        "d_model": 1536,
        "n_layers": 12,
        "lr": 4e-4,
        "batch_size": 8,
        "batches_per_step": 3,
        "seed": 98742,
        # 'checkpoint_every_tokens':5*10**7,
        "use_checkpoint_schedule": True,
        "debug": False,
        "debug_batch": False,
        "debug_overfit": False,
        "normalization": "LN",  # 'LN' 'RMS' or None
        # "max_tokens": 15 * 10 ** 9,
        "max_tokens": 2e8,
        "version": 29,
        "use_float16": False,
        "use_bfloat16": False,
        "save_checkpoints_to_bfloat16": True,
        "use_bfloat16_matmul": True,
        "right_multiply_matrices": True,
        # 'n_heads':8,
        "d_head": 128,
        "n_ctx": 1024,
        "d_vocab": 50278,
        # 'factor_size':256,
        "betas": (0.9, 0.99),
        "weight_decay": 0.01,
        "dataset_name": "the_pile",
        # "dataset_name": "wikipedia",
        # "dataset_subset_name": "20220301.en",
        "grad_norm_clip": 1.0,
        "use_attn_result": False,
        "n_devices": torch.cuda.device_count(),
        "act_fn": "SoLU",
        "use_pos_resid": True,
        "attn_only": False,
        "ln_eps": 1e-5,
        "lr_schedule": "cosine_warmup",
        # "warmup_tokens": 25 * 10 ** 7,
        "warmup_tokens": 2e6,
        "factored_embed": False,
        "train_loss_ewma_beta": 0.99,
        "shuffled_data": True,
        # 'W_O_init_scale':True,
    }
    accelerator.print(cfg)
    # accelerato(cfg)
    # print()
    cfg["n_heads"] = cfg["d_model"] // cfg["d_head"]
    cfg["d_mlp"] = 4 * cfg["d_model"]
    cfg["tokens_per_step"] = cfg["batch_size"] * cfg["n_ctx"] * cfg["batches_per_step"]
    cfg["max_steps"] = cfg["max_tokens"] // cfg["tokens_per_step"]
    cfg["warmup_steps"] = cfg["warmup_tokens"] // cfg["tokens_per_step"]
    # cfg['checkpoint_every'] = cfg['checkpoint_every_tokens']//cfg['tokens_per_step']
    if cfg["debug"] and not cfg["debug_overfit"]:
        print("Old max steps:", cfg["max_steps"])
        cfg["max_steps"] = 20
    cfg["n_params"] = 12 * cfg["n_layers"] * cfg["d_model"] ** 2
    # cfg['warmup_steps']=cfg['warmup_tokens']//cfg['tokens_per_step']
    # pprint(cfg)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    accelerator.print(f"Num params: {12*cfg['n_layers']*cfg['d_model']**2:,}")
    accelerator.print()
    return cfg


def cuda_memory():
    print(
        [
            torch.cuda.memory_allocated(f"cuda:{i}") / 1e9
            for i in range(torch.cuda.device_count())
        ]
    )


def get_corner(tensor, n=2):
    # Prints the top left corner of the tensor
    if len(tensor.shape) == 0:
        return tensor
    elif len(tensor.shape) == 1:
        return tensor[:n]
    elif len(tensor.shape) == 2:
        return tensor[:n, :n]
    elif len(tensor.shape) == 3:
        return tensor[:n, :n, :n]
    elif len(tensor.shape) == 4:
        return tensor[:n, :n, :n, :n]
    elif len(tensor.shape) == 5:
        return tensor[:n, :n, :n, :n, :n]
    elif len(tensor.shape) == 6:
        return tensor[:n, :n, :n, :n, :n, :n]
    else:
        # I never need tensors of rank > 6
        raise ValueError(f"Tensor of shape {tensor.shape} is too big")


def to_numpy(tensor, flat=False):
    if (type(tensor) != torch.Tensor) and (
        type(tensor) != torch.nn.parameter.Parameter
    ):
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def save_to_bfloat16(model, file_name):
    sd = model.state_dict()
    torch.save({k: v.to(torch.bfloat16) for k, v in sd.items()}, file_name)
    print("Saved model as bfloat16 to", file_name)


# %%
