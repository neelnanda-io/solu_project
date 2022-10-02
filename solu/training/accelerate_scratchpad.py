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
from rich import print# as rprint

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
        
        model_cfg = EasyTransformerConfig.from_dict(model_cfg_dict)
        if cfg_dict['debug']:
            rprint(training_cfg_dict)
            rprint(model_cfg_dict)
        return cls(model_cfg = model_cfg, **training_cfg_dict)

cfg = TrainingConfig.from_dict(
    n_layers = 2,
    apply_anthropic_hyper_params = True,
    act_fn='solu_ln',
    tokenizer_name = "EleutherAI/gpt-neox-20b",
    lr=1e-4,
    n_ctx=1024
)
rprint(cfg)

# %%
model = EasyTransformer.from_config(cfg.model_cfg)
rprint(model)

# %%
