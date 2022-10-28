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
import tqdm.auto as tqdm

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

from easy_transformer.evals import evaluate, induction_loss
from easy_transformer.utils import lm_cross_entropy_loss
from easy_transformer import EasyTransformer

INITIALIZATION_DIR = Path("/workspace/solu_project/initialization")

DEFAULT_CFG = {
    "model_name": "gpt2",
    "act_fn": "solu_ln",
    "lr": 1e-4,  # Effective this / d_model
    "batch_size_per_device": 16, # This is batch_size_per_device
    "batches_per_step": 1,
    "seed": -1,
    "save_checkpoints": True,
    "debug": False,
    "debug_batch": False,
    "max_tokens": 5*10**9,
    "version": -1,
    "tokenizer_name": "gpt2",
    # "d_vocab": 50278,
    # "tokenizer_name": "EleutherAI/gpt-neox-20b",
    "betas": (0.9, 0.99),
    "weight_decay": 0.1,
    "dataset_name": "c4_gpt2",
    "grad_norm_clip": 1.0,
    "n_devices": torch.cuda.device_count(),
    "lr_schedule": "cosine_warmup",
    "warmup_tokens": 10**8,
    "train_loss_ewma_beta": 0.99,
    "truncate_tokens": 10 ** 12,
    # "truncate_tokens": 10 ** 6,
    "log_interval": 50,
    "control": 1.,
    "n_ctx": 1024,
}

def create_cfg(parsed_args):
    cfg = dict(DEFAULT_CFG)
    for key in parsed_args:
        if key not in cfg:
            print("KEY NOT IN CFG!", key)
            raise ValueError
    cfg.update(parsed_args)

    cfg["version"] = max(solu_utils.solu_get_prev_versions()) + 1

    cfg["tokens_per_step"] = cfg["batch_size_per_device"] * \
        cfg["n_ctx"] * cfg["batches_per_step"] * cfg["n_devices"]
    
    cfg["batch_size"] = cfg["batch_size_per_device"] * cfg["n_devices"]

    cfg["max_steps"] = cfg["max_tokens"] // cfg["tokens_per_step"]
    cfg["warmup_steps"] = cfg["warmup_tokens"] // cfg["tokens_per_step"]

    if cfg["debug"]:
        cfg["truncate_tokens"] = 5 * 10**6
    
    if cfg["seed"]==-1:
        cfg["seed"] = random.randint(0, 2**20 - 1)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    set_seed(cfg["seed"])
    return cfg

def get_max_batch_size(cfg):
    model = make_model(cfg)
    # init_weights(model, cfg)
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
    print("Loaded tokenizer:", tokenizer)
    return tokenizer


def create_dataset(cfg, tokenizer):
    data = datasets.load_from_disk("/workspace/data/c4_train_30_tokens_gpt2.hf")
    if cfg["debug"]:
        print(data)
    print("Loaded dataset")
    data = data.with_format("torch")
    print("Convert to torch")
    data = data.shuffle(seed=cfg['seed'])
    print("Shuffled")
    batch_size = cfg["batch_size_per_device"] * cfg["n_devices"]

    data_loader = DataLoader(data, num_workers=8, batch_size=batch_size)
    print("Made data loader")
    print("Created C4 GPT2 dataset")
    return data_loader

def test(args):
    # from neel.imports import *; from solu.training.train_model_ddp import *
    cfg = create_cfg(args)
    tokenizer = init_tokenizer()

    model = Transformer(cfg)
    data_loader = create_dataset(cfg,tokenizer)
    data = next(iter(data_loader))['text'][:4]
    # init_weights(model, cfg)
    loss = model(data)
    return cfg, model, tokenizer, data_loader, data, loss

def make_model_name(cfg):
    return f"v{cfg['version']}_{cfg['model_name']}_fine_tuned_solu"

def make_model(cfg):
    original_gpt2 = EasyTransformer.from_pretrained(cfg["model_name"], fold_ln=False, center_writing_weights=False, center_unembed=False, device='cpu')
    model_cfg = copy.deepcopy(original_gpt2.cfg)
    model_cfg.act_fn = cfg["act_fn"]
    model = EasyTransformer(model_cfg, original_gpt2.tokenizer)
    model.load_state_dict(original_gpt2, strict=False)
    return model

# %%
def main(ipython_args=None):
    if MODE != "wandb":
        parser = argparse.ArgumentParser()
        for key, value in DEFAULT_CFG.items():
            if type(value)==bool:
                # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
                if value:
                    parser.add_argument(f"--{key}", action="store_false")
                else:
                    parser.add_argument(f"--{key}", action="store_true")

            else:
                parser.add_argument(f"--{key}", type=type(value), default=value)

        args = parser.parse_args()
        cfg = create_cfg(vars(args))
    else:
        run = wandb.init(project="solu",
        entity="mechanistic-interpretability", 
        config=DEFAULT_CFG)
    
        print(run.config)
        cfg = dict(run.config)
        print("Setting run config to wandb")
        print(run.config["n_layers"])
        cfg = create_cfg(cfg)

    set_seed(cfg["seed"])
    
    print(json.dumps(cfg, indent=2))

    if MODE!="wandb":
        model_name = make_model_name(cfg)
        save_dir = Path(f"/workspace/solu_project/saved_models/{model_name}")
        checkpoint_dir = save_dir/"checkpoints"
        save_dir.mkdir(exist_ok=True, parents=False)
        checkpoint_dir.mkdir(exist_ok=True, parents=False)
        
        wandb.init(project="solu",
        entity="mechanistic-interpretability", 
        config=cfg, name=model_name)

    model = make_model(cfg)
    tokenizer = model.tokenizer
    data_loader = create_dataset(cfg, tokenizer)
    
    if not cfg["debug"] and MODE!="wandb":
        torch.save(model.state_dict(), save_dir/"model_init.pth")
        if cfg["store_init"]:
            torch.save(model.state_dict(), INITIALIZATION_DIR/f"{model_name}.pth")
        with open(save_dir/"config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    model.to("cuda")
    
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
                "weight_decay": cfg["weight_decay"], "lr":cfg["lr"]/(model.cfg.d_model/256)},
            {"params": param_groups["vector"], "weight_decay": 0.0, "lr":cfg["lr"]},
        ]
        optimizer = torch.optim.AdamW(optim_groups)
        print(optimizer)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
        print(scheduler)

    if cfg["save_checkpoints"]:
        schedule = solu_utils.SaveSchedule(
            cfg["max_tokens"],
            cfg["tokens_per_step"],
        )
        
    parallel_model = torch.nn.DataParallel(model)

    print("Training begins!")

    step = 0
    start_time = time.time()
    loss_ewma = torch.tensor(9., device='cuda')
    total_tokens = 0
    running_loss = torch.tensor(0., device='cuda')
    
    data_iter = iter(data_loader)
    data = (next(data_iter))
    print(data['tokens'][:3, :100])
    print(tokenizer.batch_decode(data['tokens'][:3, :100]))
    for c in tqdm.tqdm(range(cfg['max_steps'])):
        while True:
            try:
                batch = next(data_iter)
                break
            except Exception as e:
                print("There was an error in data iter:", e)
                continue
        
        batch = batch["tokens"]
        
        loss = parallel_model(batch).mean() / cfg["batches_per_step"]

        if c<2:
            print(loss.item())
        loss.backward()
        if c < 3:
            print(batch.shape)
        
        running_loss += loss.detach()
        total_tokens += cfg["tokens_per_step"]

        if (c + 1) % cfg["batches_per_step"] == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["grad_norm_clip"])
            optimizer.step()
            if cfg["lr_schedule"] is not None:
                scheduler.step()
            optimizer.zero_grad()

            if (
                MODE!="wandb"
                and schedule.step()
                and cfg["save_checkpoints"]
                and not cfg["debug"]
            ):
                print(
                    f"Saved the model! Step: {step}. Frac of way through training: {schedule.schedule[schedule.next_save_point-1]}"
                )
                if not cfg["debug"]:
                    torch.save(model.state_dict(), checkpoint_dir/f"tokens_{total_tokens:0>12}.pth")
                if step > cfg["max_steps"] * 0.05:
                    print(f"Saving optimizer and scheduler checkpoints at step {step}")
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

            if step % cfg["log_interval"] == 0:
                log_dict = {
                        "loss": running_loss.item(),
                        "loss_ewma": loss_ewma.item(),
                        "elapsed": time.time() - start_time,
                        "total_tokens": total_tokens,
                        "c": c,
                        "scheduled_lr": scheduler.get_last_lr()[0],
                        "induction_loss": induction_loss(model, tokenizer, batch_size=4, subseq_len=300).item(),
                    }
                print(json.dumps(log_dict, indent=2))
                wandb.log(
                    log_dict,
                    step=step,
                )
            running_loss = running_loss * 0.
            step += 1
            
            if step >= cfg["max_steps"] or step * cfg["tokens_per_step"] >= cfg["truncate_tokens"]:
                print("Step limit reached at step", step)
                break
        if cfg["debug"] and c < 3:
            print("Batch shape:", batch.shape)
            print(batch[0, :100])
            print(tokenizer.decode(batch[0])[:200])

        del loss

    print(f"Finished training! Train Loss EWMA: {loss_ewma}")

    eval_losses = evaluate(parallel_model, 200, cfg['batch_size'], tokenizer)
    print(f"Eval Loss: {eval_losses}")

    wandb.log(eval_losses, step=step)

    if not cfg["debug"] and MODE != "wandb":
        torch.save(model.state_dict(), save_dir/f"model_final.pth")
        # if total_tokens > 5e9:
        #     solu_utils.move_folder_to_hub(model_name, just_final=True)
        wandb.finish()
    
# %%

""" 
Test string:

cfg = create_cfg(dict(n_layers=4, batch_size_per_device=32))
model = Transformer(cfg)
tokenizer = init_tokenizer(cfg)
model.tokenizer = tokenizer
import easy_transformer.evals as evals
model.load_state_dict(torch.load("/workspace/solu_project/saved_models/v145_4L512W_solu_repo/model_final.pth"))
model.cuda()
parallel_model = torch.nn.DataParallel(model)
evaluate(parallel_model, truncate=100, batch_size=cfg['batch_size'], tokenizer=tokenizer)
"""

# if __name__=="__main__":
#     main()

SWEEP_PARAMS = {
            'n_layers': {
                'value': 2
            },
            'save_checkpoints': {
                'value': False
            },
            "seed": {
                "values": [124, 456, 789],
            },
            "truncate_tokens": {"value": 25 * 10**8},
            "lr_hidden": {
                "distribution": "log_uniform_values",
                "min":4e-4,
                "max":1e-2,
            },
            "lr_vector": {
                "distribution": "log_uniform_values",
                "min":4e-4,
                "max":3e-2,
            },
            "initializer_scale_global": {
                "distribution": "log_uniform_values",
                "min":1e-1,
                "max":1e1,
            },
            "neuron_temp": {
                "distribution": "log_uniform_values",
                "min":0.4,
                "max":4.,
            },
            "neuron_scale": {
                "distribution": "log_uniform_values",
                "min":1e-1,
                "max":1e1,
            },
            "control": {
                "distribution": "uniform",
                "min":0.,
                "max":1.,
            },
        }
DO_WANDB_SWEEP = False
# RESUME_WANDB_SWEEP = True
# MODE = "wandb"
MODE = "dp"
if __name__=="__main__":
    if DO_WANDB_SWEEP:
        sweep_config = {
            'method': 'random',
            'parameters': SWEEP_PARAMS,
            }
        
        sweep_id = wandb.sweep(sweep_config, project="solu")
        print("SWEEP ID:", sweep_id)
        wandb.agent(sweep_id, function=main, count=1000)
    else:
        main()

# %%