# %%

from pprint import pprint
from accelerate import notebook_launcher
from accelerate.utils import set_seed, write_basic_config
from accelerate import Accelerator
import accelerate
import wandb
import datasets
from transformers import AutoTokenizer
import json
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import itertools
import copy
import collections
import gc
import pandas as pd
from functools import *
from torch.utils.data import DataLoader
import plotly.graph_objects as go
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
import argparse


pio.renderers.default = "vscode"
import solu.utils as solu_utils
from solu.transformer import Transformer


INITIALIZATION_DIR = Path("/workspace/solu_project/initialization")

DEFAULT_CFG = {
    "n_layers": -1,
    "d_model": -1, # 128 * n_layers
    "d_mlp": -1, # 4 * d_model
    "d_head": 64, # 64
    "n_heads": -1, # d_model//d_head
    "lr_hidden": 3e-3,  # Effective this / d_model
    "lr_vector": 1.5e-3, 
    "batch_size_per_device": 32, # This is batch_size_per_device
    "batches_per_step": 1,
    "seed": -1,
    "save_checkpoints": True,
    "debug": False,
    "debug_batch": False,
    "normalization": "LN",  # 'LN' 'RMS' or None
    "max_tokens": 10*10**9,
    "version": -1,
    "use_bfloat16_matmul": True,
    "n_ctx": 1024,
    "d_vocab": 48262,
    "tokenizer_name": "NeelNanda/gpt-neox-tokenizer-digits",
    "betas": (0.9, 0.99),
    "weight_decay": 0.05,
    "dataset_name": "c4+code",
    "grad_norm_clip": 1.0,
    "n_devices": torch.cuda.device_count(),
    "act_fn": "solu_ln",
    "shortformer_pos": False,
    "attn_only": False,
    "ln_eps": 1e-5,
    "lr_schedule": "cosine_warmup",
    "warmup_tokens": 3*10**8,
    "train_loss_ewma_beta": 0.99,
    "truncate_tokens": 10**12,
    "log_interval": 50,
    "initializer_scale_global": 1.,
    "initializer_scale_hidden": 0.0125, # This / sqrt(d_model/256), used for attn and neurons
    "initializer_scale_embed": 1e-2, # This, constant
    "initializer_scale_unembed": 0.05, # Set to this / (d_model/256)
    "use_acc": False,
    "weight_init_scheme": "mup",
    "fixed_init": "", # The name of the saved initialization file
    "store_init": False, # Whether to store the initialization for use in future runs.
}

def create_cfg(parsed_args):
    cfg = dict(DEFAULT_CFG)
    for key in parsed_args:
        if key not in cfg:
            print("KEY NOT IN CFG!", key)
            raise ValueError
    cfg.update(parsed_args)

    cfg["version"] = max(solu_utils.solu_get_prev_versions()) + 1

    if cfg["d_model"]==-1:
        cfg["d_model"] = 128 * cfg["n_layers"]

    if cfg["n_heads"]==-1:
        cfg["n_heads"] = cfg["d_model"] // cfg["d_head"]
    if cfg["d_mlp"]==-1:
        cfg["d_mlp"] = 4 * cfg["d_model"]
    cfg["tokens_per_step"] = cfg["batch_size_per_device"] * \
        cfg["n_ctx"] * cfg["batches_per_step"] * cfg["n_devices"]

    cfg["max_steps"] = cfg["max_tokens"] // cfg["tokens_per_step"]
    cfg["warmup_steps"] = cfg["warmup_tokens"] // cfg["tokens_per_step"]

    if cfg["debug"]:
        cfg["truncate_tokens"] = 5 * 10**6
    if not cfg['attn_only']:
        # Formula for number of non-embedding parameters in standard parametrization
        cfg["n_params"] = 12 * cfg["n_layers"] * cfg["d_model"] ** 2
    else:
        # Formula for number of non-embedding parameters in standard parametrization
        cfg["n_params"] = 3 * cfg["n_layers"] * cfg["d_model"] ** 2
    
    if cfg["seed"]==-1:
        cfg["seed"] = random.randint(0, 2**32 - 1)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    set_seed(cfg["seed"])
    return cfg

def get_max_batch_size(cfg):
    model = Transformer(cfg, None)
    init_weights(model, cfg)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    parallel_model = torch.nn.DataParallel(model)
    batch_scan = list(range(8, 16)) + list(range(16, 32, 2)) + list(range(32, 64, 4)) + list(range(64, 128, 8))
    best_batch_size = 0
    for batch_size in tqdm.tqdm(batch_scan):
        try:
            for i in range(2):
                rand_tokens = torch.randint(100, 10000, (batch_size*torch.cuda.device_count(), cfg["n_ctx"])).cuda()
                print("Shape:", rand_tokens.shape)
                loss = parallel_model(rand_tokens).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print("Successfully tested batch size", batch_size)
            best_batch_size = batch_size
        except Exception as e:
            print("Error:", e)
            return best_batch_size
# %%

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

def init_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    return tokenizer


def create_dataset(cfg):

    data = datasets.concatenate_datasets(
        [
            datasets.load_from_disk("/workspace/data/c4_train_160_tokens.hf"),
            datasets.load_from_disk("/workspace/data/codeparrot_train_tokens.hf"),
            #! TODO Fix after!
            # datasets.load_from_disk("/workspace/data/c4_train_1_tokens.hf"),
            # datasets.load_from_disk("/workspace/data/codeparrot_valid_tokens.hf"),
        ])
    if cfg["debug"]:
        print(data)
    data = data.with_format("torch")
    data = data.shuffle(seed=cfg['seed'])
    if cfg["use_acc"]:
        batch_size = cfg["batch_size_per_device"]
    else:
        batch_size = cfg["batch_size_per_device"] * cfg["n_devices"]

    data_loader = DataLoader(data, num_workers=8, batch_size=batch_size)
    return data_loader

def init_weights(model, cfg):
    if cfg["fixed_init"]:
        print("Using custom initialization from", cfg["fixed_init"])
        init_state_dict = torch.load(INITIALIZATION_DIR/f"{cfg['fixed_init']}.pth")
        current_state_dict = model.state_dict()
        filtered_state_dict = {k:v for k, v in init_state_dict.items() if k in current_state_dict}
        model.load_state_dict(filtered_state_dict, strict=True)
    else:
        # Using mu-P factorization
        global_scale = cfg["initializer_scale_global"]
        # Set in my hyper-param sweep over a 2L256W model
        base_d_model = 256
        for name, param in model.named_parameters():
            if "W_" in name:
                if name.endswith("W_U"):
                    scale = cfg["initializer_scale_unembed"] * global_scale / (cfg["d_model"]/base_d_model)
                    torch.nn.init.normal_(param, std=scale)
                elif name.endswith("W_E") or name.endswith("W_pos"):
                    scale = cfg["initializer_scale_embed"] * global_scale
                    torch.nn.init.normal_(param, std=scale)
                elif (
                    name.endswith("W_K") or
                    name.endswith("W_Q") or
                    name.endswith("W_V") or
                    name.endswith("W_O") or
                    name.endswith("W_in") or
                    name.endswith("W_out")
                    ):
                    scale = cfg["initializer_scale_hidden"] * global_scale / np.sqrt(cfg["d_model"]/base_d_model)
                    torch.nn.init.normal_(param, std=scale)
                    if name.endswith("W_out"):
                        param.data = param.data / 2
                else:
                    ValueError(f"Unknown weight name {name}")

def test(args):
    # from neel.imports import *; from solu.training.train_model_ddp import *
    cfg = create_cfg(args)
    tokenizer = init_tokenizer()

    model = Transformer(cfg)
    data_loader = create_dataset(cfg)
    data = next(iter(data_loader))['text'][:4]
    init_weights(model, cfg)
    loss = model(data)
    return cfg, model, tokenizer, data_loader, data, loss

def make_model_name(cfg):
    if cfg['attn_only']:
        leaf = 'attn_only'
    elif cfg['act_fn']=='solu_ln':
        leaf = 'solu'
    elif cfg['act_fn']=='gelu':
        leaf = 'gelu'
    else:
        raise ValueError(f"Invalid config for model name: {cfg}")
    return f"v{cfg['version']}_{cfg['n_layers']}L{cfg['d_model']}W_{leaf}"

# %%
def main(ipython_args=None):

    parser = argparse.ArgumentParser()
    for key, value in DEFAULT_CFG.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

    args = parser.parse_args()
    cfg = create_cfg(vars(args))
    set_seed(cfg["seed"])
    accelerator = Accelerator(gradient_accumulation_steps=cfg["batches_per_step"])
    if accelerator.num_processes > 1:
        accelerator.print("Using accelerate!")
        cfg["use_acc"] = True 
    
    
    print("Is main process", accelerator.is_main_process)
    accelerator.print("initialized accelerator")
    accelerator.print(json.dumps(cfg, indent=2))

    if accelerator.is_main_process:
        model_name = make_model_name(cfg)
        save_dir = Path(f"/workspace/solu_project/saved_models/{model_name}")
        checkpoint_dir = save_dir/"checkpoints"
        save_dir.mkdir(exist_ok=True, parents=False)
        checkpoint_dir.mkdir(exist_ok=True, parents=False)
        
        wandb.init(project="solu",
        entity="mechanistic-interpretability", 
        config=cfg, name=model_name)

    data_loader = create_dataset(cfg)

    tokenizer = init_tokenizer(cfg)

    model = Transformer(cfg)

    if cfg["weight_init_scheme"]=="mup":
        print("Mup init scheme")
        init_weights(model, cfg)
    else:
        raise ValueError(f"Bad init scheme: {cfg['weight_init_scheme']}")
        
    
    if accelerator.is_main_process and not cfg["debug"]:
        torch.save(model.state_dict(), save_dir/"model_init.pth")
        if cfg["store_init"]:
            torch.save(model.state_dict(), INITIALIZATION_DIR/f"{model_name}.pth")
        with open(save_dir/"config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    model.to(accelerator.device)
    
    if cfg["lr_schedule"] is not None:

        def lr_schedule(step):
            if step < cfg["warmup_steps"]:
                return (1e-4 + (1 - 1e-4) * step / cfg["warmup_steps"])
                
            else:
                return 0.55 + 0.9 * 0.5 * np.cos(
                    np.pi
                    * (step - cfg["warmup_steps"])
                    / (cfg["max_steps"] - cfg["warmup_steps"])
                )

        param_groups = {"hidden+unembed": [], "vector": []}
        for name, param in model.named_parameters():
            suffix = name.split(".")[-1]
            if suffix in ["W_K", "W_V", "W_Q", "W_O", "W_U", "W_out", "W_in"]:
                param_groups["hidden+unembed"].append(param)
            elif suffix in ["W_E", "W_pos"]:
                param_groups["vector"].append(param)
            elif suffix=="w" or "b" in suffix:
                param_groups["vector"].append(param)
            else:
                print("Invalid weight name:", name)
                raise ValueError
        optim_groups = [
            {"params": param_groups["hidden+unembed"],
                "weight_decay": cfg["weight_decay"], "lr":cfg["lr_hidden"]/(cfg["d_model"]/256)},
            {"params": param_groups["vector"], "weight_decay": 0.0, "lr":cfg["lr_vector"]},
        ]
        optimizer = torch.optim.AdamW(optim_groups)
        accelerator.print(optimizer)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    if accelerator.is_main_process and cfg["save_checkpoints"]:
        schedule = solu_utils.SaveSchedule(
            cfg["max_tokens"],
            cfg["tokens_per_step"],
        )

    model, optimizer, data_loader = accelerator.prepare(
        model, optimizer, data_loader)

    if not cfg["use_acc"]:
        parallel_model = torch.nn.DataParallel(model)

    accelerator.print("Training begins!")

    step = 0
    start_time = time.time()
    loss_ewma = torch.tensor(9., device=accelerator.device)
    total_tokens = 0
    running_loss = torch.tensor(0., device=accelerator.device)
    for c, batch in tqdm.tqdm(enumerate(data_loader)):
        with accelerator.accumulate(model):
            batch = batch["tokens"]
            
            if not cfg["use_acc"]:
                loss = parallel_model(batch).mean() / cfg["batches_per_step"]
            else:
                loss = model(batch)

            if c<2:
                accelerator.print(loss)
            accelerator.backward(loss)
            if c < 3 and accelerator.is_main_process:
                accelerator.print(batch.shape)
            
            running_loss += loss.detach()
            total_tokens += cfg["tokens_per_step"]

            if (c + 1) % cfg["batches_per_step"] == 0:
                accelerator.clip_grad_norm_(
                    model.parameters(), cfg["grad_norm_clip"])
                optimizer.step()
                if cfg["lr_schedule"] is not None:
                    scheduler.step()
                optimizer.zero_grad()

                if (
                    accelerator.is_main_process
                    and schedule.step()
                    and cfg["save_checkpoints"]
                    and not cfg["debug"]
                ):
                    accelerator.print(
                        f"Saved the model! Step: {step}. Frac of way through training: {schedule.schedule[schedule.next_save_point-1]}"
                    )
                    if not cfg["debug"]:
                        
                        torch.save(model.state_dict(), checkpoint_dir/f"tokens_{total_tokens:0>12}.pth")
                if accelerator.is_main_process and (step+1) % 500 == 0:
                    accelerator.print(f"Saving optimizer and scheduler checkpoints at step {step}")
                    torch.save(
                        optimizer.state_dict(
                        ), save_dir/"optimizer_state_dict.pth")

                    torch.save(
                        scheduler.state_dict(),
                        save_dir/"scheduler_state_dict.pth",
                    )
                loss_ewma = loss_ewma * cfg["train_loss_ewma_beta"] + running_loss * (
                    1 - cfg["train_loss_ewma_beta"]
                )
                accelerator.wait_for_everyone()
                if accelerator.is_main_process and step % cfg["log_interval"] == 0:
                    loss_ewma = accelerator.reduce(loss_ewma, "mean")
                    log_dict = {
                            "loss": running_loss.item(),
                            "loss_ewma": loss_ewma.item(),
                            "elapsed": time.time() - start_time,
                            "total_tokens": total_tokens,
                            "c": c,
                            "scheduled_lr": scheduler.get_last_lr()[0],
                        }
                    accelerator.print(json.dumps(log_dict, indent=2))
                    wandb.log(
                        log_dict,
                        step=step,
                    )
                running_loss = running_loss * 0.
                step += 1
                
                if step >= cfg["max_steps"] or step * cfg["tokens_per_step"] >= cfg["truncate_tokens"]:
                    accelerator.print("Step limit reached at step", step)
                    break
            if cfg["debug"] and c < 3 and accelerator.is_main_process:
                accelerator.print("Batch shape:", batch.shape)
                accelerator.print(batch[0, :100])
                accelerator.print(tokenizer.decode(batch[0])[:200])

            del loss

    accelerator.print(f"Finished training! Train Loss EWMA: {loss_ewma}")

    if not cfg["debug"] and accelerator.is_main_process:
        torch.save(model.state_dict(), save_dir/f"model_final.pth")
        solu_utils.move_folder_to_hub(model_name, just_final=True)
        wandb.finish()
    
# %%
if __name__=="__main__":
    main()