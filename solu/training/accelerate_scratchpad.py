# %%
# Imports - 
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
from easy_transformer import EasyTransformer
from rich import print as rprint

# %%
from IPython import get_ipython
try:
    ipython = get_ipython()
    # Code to automatically update the EasyTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    import plotly.io as pio
    pio.renderers.default = "vscode"
    IS_IPYTHON = True
except:
    IS_IPYTHON = False
  

# %%
@dataclass
class TrainingConfig:
    apply_anthropic_hyper_params: bool
    lr: float
    batch_size: int

    batches_per_step: int = 1
    seed: int = 12345
    debug: bool = True
    initializer_scale: float = 1.

    model_cfg: EasyTransformerConfig = None
    
    @classmethod
    def from_dict(cls, **cfg_dict):
        model_config_keys = EasyTransformerConfig.__dataclass_fields__.keys()
        
        model_cfg_dict = {k:v for k, v in cfg_dict.items() if k in model_config_keys}
        training_cfg_dict = {k:v for k, v in cfg_dict.items() if k not in model_config_keys}

        if training_cfg_dict['apply_anthropic_hyper_params']:
            n_layers = model_cfg_dict['n_layers']
            model_cfg_dict['d_model'] = n_layers * 128
            model_cfg_dict['d_mlp'] = 4 * model_cfg_dict['d_model']
            model_cfg_dict['d_head'] = 64
            assert model_cfg_dict['d_model'] % model_cfg_dict['d_head'] == 0, f"d_head: {model_cfg_dict['d_head']} is not a divisor of d_model: {model_cfg_dict['d_model']}"
            model_cfg_dict['n_heads'] = model_cfg_dict['d_model']//model_cfg_dict['d_head']
        model_cfg_dict['attn_scale_full'] = True
        model_cfg = EasyTransformerConfig.from_dict(model_cfg_dict)
        # if cfg_dict['debug']:
        #     rprint(training_cfg_dict)
        #     rprint(model_cfg_dict)
        return cls(model_cfg = model_cfg, **training_cfg_dict)

cfg = TrainingConfig.from_dict(
    n_layers = 2,
    apply_anthropic_hyper_params = True,
    act_fn='solu_ln',
    tokenizer_name = "EleutherAI/gpt-neox-20b",
    lr=1e-3,
    n_ctx=1024,
    batch_size=2,
)
rprint(cfg)

# %%
model = EasyTransformer.from_config(cfg.model_cfg)
rprint(model)

# %%

# What do?
"""
- Load data
- Make optimizer & scheduler
- Setup logging
- Setup training loop
- Setup saving code + structure - config to JSON, model weights, optimizer state, scheduler state, training state
- 
"""

def load_data(cfg):
    data = datasets.concatenate_datasets([datasets.load_from_disk(Path.home()/f"data/pile_0{i}.hf") for i in range(3)])
    data = data.with_format("torch")
    data.shuffle(seed=cfg.seed)
    print(data)
    data_loader = DataLoader(data, num_workers=8, batch_size=cfg.batch_size)
    return data_loader

data_loader = load_data(cfg)
# %%
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_layers", type=int)
    for key, value in DEFAULT_CONFIG.items():
        if key != "n_layers":
            parser.add_argument(f"--{key}", type=type(value), default=value)

    if IN_IPYTHON:
        args = parser.parse_args(["1"])
    else:
        args = parser.parse_args()

    cfg = TrainingConfig.from_dict(**vars(args))

    print(
        f"Config for {cfg['n_layers']}L v{cfg['version']} with {cfg['n_params']/1e6:.2f}M params")
    for key in (cfg.keys()):
        print(f"{key}: {cfg[key]}")
    print(
        f"Config for {cfg['n_layers']}L v{cfg['version']} with {cfg['n_params']/1e6:.2f}M params")


    model = EasyTransformer.from_config(cfg.model_cfg)
    for name, param in model.named_parameters():
        scale = 1600/cfg.d_model
        if "W_" in name:
            if name.endswith("W_E") or name.endswith("W_pos"):
                param.data.normal_(mean=0.0, std=cfg.initializer_scale)
            elif name.endswith("W_U"):
                param.data.normal_(mean=0.0, std=(0.02*scale)**2 * cfg.initializer_scale)
            else:
                param.data.normal_(mean=0.0, std=(0.02*scale) * cfg.initializer_scale)

    accelerator = Accelerator(
    mixed_precision='bf16', gradient_accumulation_steps=cfg.batches_per_step)
    
    

