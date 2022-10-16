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


INITIALIZATION_DIR = Path("/workspace/solu_project/initialization")

DEFAULT_CFG = {
    "n_layers": -1,
    "d_model": -1, # 128 * n_layers
    "d_mlp": -1, # 4 * d_model
    "d_head": 64, # 64
    "n_heads": -1, # d_model//d_head
    "lr_hidden": 2e-3,  # Effective this / d_model
    "lr_vector": 1.5e-3, 
    "batch_size_per_device": 32, # This is batch_size_per_device
    "batches_per_step": 1,
    "seed": -1,
    "save_checkpoints": True,
    "debug": False,
    "debug_batch": False,
    "normalization": "LN",  # 'LN' 'RMS' or None
    "max_tokens": 15*10**9,
    "version": -1,
    "use_bfloat16_matmul": True,
    "n_ctx": 1024,
    "d_vocab": 48262,
    # "tokenizer_name": "NeelNanda/gpt-neox-tokenizer-digits",
    "tokenizer_name": "EleutherAI/gpt-neox-20b",
    "betas": (0.9, 0.99),
    "weight_decay": 0.05,
    # "dataset_name": "c4+code",
    "dataset_name": "the_pile",
    "grad_norm_clip": 1.0,
    "n_devices": torch.cuda.device_count(),
    "act_fn": "solu_ln",
    "shortformer_pos": False,
    "attn_only": False,
    "ln_eps": 1e-5,
    "lr_schedule": "cosine_warmup",
    "warmup_tokens": 3*10**8,
    "train_loss_ewma_beta": 0.99,
    "truncate_tokens": 10 ** 12,
    "log_interval": 50,
    "initializer_scale_global": 1.,
    "initializer_scale_hidden": 0.02, # This / sqrt(d_model/256), used for attn and neurons
    "initializer_scale_embed": 1e-1, # This, constant
    "initializer_scale_unembed": 0.02, # Set to this / (d_model/256)
    "neuron_scale": 0.5,
    "neuron_temp": 2.,
    "use_acc": False,
    "weight_init_scheme": "mup",
    "fixed_init": "", # The name of the saved initialization file
    "store_init": False, # Whether to store the initialization for use in future runs.
    "control": 1.,
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
    model = Transformer(cfg)
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
    print("Loaded tokenizer:", tokenizer)
    return tokenizer


def create_dataset(cfg, tokenizer):
    if cfg['dataset_name']=='c4+code':
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
        print("Created C4 + CodeParrot dataset")
        return data_loader
    else:
        return create_pile_dataset(cfg, tokenizer)

def create_pile_dataset(cfg, tokenizer):
    seq_len = cfg['n_ctx']
    def tokenize(examples):
        start_time = time.time()
        texts = examples['text']
        full_text = tokenizer.eos_token.join(texts)
        div = 20
        length = len(full_text)//div
        text_list = [full_text[i*length:(i+1)*length] for i in range(div)]
        tokens = tokenizer(text_list, return_tensors='np', padding=True)['input_ids'].flatten()
        tokens = tokens[tokens!=tokenizer.pad_token_id]
        # print(len(text_list), len(text_list[0]))
        # print(tokens.shape)
        n = len(tokens)
        curr_batch_size = n//(seq_len-1)
        tokens = tokens[:(seq_len-1)*curr_batch_size]
        tokens = einops.rearrange(tokens, '(batch_size seq) -> batch_size seq', batch_size=curr_batch_size, seq=seq_len-1)
        prefix = np.ones((curr_batch_size, 1), dtype=np.int64)*tokenizer.bos_token_id
        # print(tokens.shape, n, curr_batch_size, seq_len)
        return {'tokens': np.concatenate([prefix, tokens], axis=1)}# 


    randperm = np.random.permutation(28)
    print('Permutation of PILE URLs', randperm)
    pile_urls = [f"https://the-eye.eu/public/AI/pile/train/{i:0>2}.jsonl.zst" for i in randperm]
    dataset = load_dataset('json', data_files=pile_urls, streaming=True, split='train')
    dataset = dataset.remove_columns('meta')

    dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])
    dataset = dataset.with_format(type='torch')
    dataset = dataset.shuffle(seed=cfg['seed'], buffer_size=30000)
    if cfg["use_acc"]:
        batch_size = cfg["batch_size_per_device"]
    else:
        batch_size = cfg["batch_size_per_device"] * cfg["n_devices"]

    train_data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)
    print("Created PILE dataset")
    return train_data_loader

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

def test_induction(model, tokenizer):
    rep_tokens = torch.randint(100, 20000, (4, 512)).cuda()
    rep_tokens = einops.repeat(rep_tokens, "b p -> b (2 p)")
    rep_tokens[:, 0] = tokenizer.bos_token_id

    logits = model(rep_tokens, return_loss=False)

    from easy_transformer.utils import lm_cross_entropy_loss

    lps = lm_cross_entropy_loss(logits, rep_tokens, return_per_token=True)

    # line(lps)
    return (lps[:, 512:].mean())

def test_wikipedia(model, tokenizer):
    big_string = """
    The Borodino-class battlecruisers (Russian: Линейные крейсера типа «Измаил») were a group of four battlecruisers ordered by the Imperial Russian Navy before World War I. Also referred to as the Izmail class, they were laid down in late 1912[Note 1] at Saint Petersburg for service with the Baltic Fleet. Construction of the ships was delayed by a lack of capacity among domestic factories and the need to order some components from abroad. The start of World War I slowed their construction still further, as the imported components were often not delivered and domestic production was diverted into areas more immediately useful for the war effort.

    Three of the four ships were launched in 1915 and the fourth in 1916. Work on the gun turrets lagged, and it became evident that Russian industry would not be able to complete the ships during the war. The Russian Revolution of 1917 halted all work on the ships, which was never resumed. Although some consideration was given to finishing the hulls that were nearest to completion, the three furthest from completion were sold for scrap by the Soviet Union during the early 1920s. The Soviet Navy proposed to convert Izmail, the ship closest to completion, to an aircraft carrier in 1925, but the plan was cancelled after political manoeuvring by the Red Army cut funding and she was eventually scrapped in 1931.

    Design and development
    After the end of the Russo-Japanese War of 1905, the Russian Naval General Staff decided that it needed a squadron of fast armoured cruisers[Note 2][1] that could use their speed to engage the leader of an enemy's battle line, as Admiral Tōgō had done against the Russian fleet during the Battle of Tsushima. The Naval General Staff initially called for a ship with high speed (28 knots (52 km/h; 32 mph)), 305-millimetre (12 in) guns, and limited protection (a waterline belt of 190 mm or 7.5 in). The Tsar, head of the Russian government, approved construction of four such ships on 5 May 1911, but the State Duma session ended before the proposal could be voted on. Preliminary bids for the ships were solicited from private builders, but the bids proved to be very high,[1] leading to a reconsideration of the requirements. The Naval General Staff issued a new specification on 1 July 1911 for a ship with a speed of only 26.5 knots (49.1 km/h; 30.5 mph) and with armour increased to 254 mm (10 in). The armament was increased to nine 356-millimetre (14 in) guns in three non-superfiring triple-gun turrets,[2] based on a false rumour that the Germans were increasing the calibre of the guns in their battleships.[3] The Imperial Russian Navy believed that widely separating the main gun turrets and their magazines reduced the chance of a catastrophic ammunition explosion, reduced the silhouette of the ship and improved stability without superfiring turrets and their tall barbettes.[4]

    The Naval Ministry solicited new bids on 8 September from 23 shipbuilders, domestic and foreign, but only 7 responded, even after the deadline was extended by a month. Several designs were rejected for not meeting the revised criteria. In the meantime, the Artillery Section of the Main Administration of Shipbuilding had decided that it preferred a four-turret design, and new bids were solicited in May 1912 from the leading contenders from the first round of bidding.[5] The eventual winner was a design by the Admiralty Shipyard in Saint Petersburg which had the extra turret added to a new hull section inserted into the original three-turret design.[6]

    The Duma approved construction in May 1912, before the design was finalised, and allocated 45.5 million rubles for each ship. The additional gun turret and consequent increase in the size of the ships led to the ships being overbudget by about 7 million rubles each, and some money was diverted from the budget for the Svetlana-class cruiser to cover the discrepancy. Orders were placed on 18 September 1912 for a pair of ships each from the Admiralty Shipyard and the Baltic Works, also of Saint Petersburg. The first pair was to be ready for trials on 14 July 1916, and the second pair on 14 September 1916.[5][7]

    Full-scale armour trials in 1913 revealed serious weaknesses in the Borodinos' proposed protection scheme. The obsolete ironclad Chesma had been modified with armour protection identical to that used by the Gangut-class battleships, then under construction. The deck and turret-roof armour proved to be too thin, and the structure supporting the side armour was not strong enough to withstand the shock of impact from heavy shells.[8] The design of the Borodinos' armour was similar in construction to that of the Ganguts and therefore needed to be modified, which slowed construction. The Borodinos' deck armour was reinforced with extra plates and the thickness of the turret roofs was increased. To compensate for this additional weight, a planned rear conning tower was removed entirely and the thickness of the main belt was slightly reduced. Mortise and tenon joints were introduced between the armour plates along their vertical edges to better distribute the shock of a shell impact and to lessen the stress on the supporting hull structure. The launching of the first pair of ships was postponed by six months because of these changes, plus delays imposed by the many ship orders already in hand.[Note 3][8]"""

    tokens = tokenizer.encode(big_string, return_tensors='pt')[:, :1024].cuda()
    return model(tokens)

# %%
def main(ipython_args=None):
    if MODE != "wandb":
        parser = argparse.ArgumentParser()
        for key, value in DEFAULT_CFG.items():
            parser.add_argument(f"--{key}", type=type(value), default=value)

        args = parser.parse_args()
        cfg = create_cfg(vars(args))
        accelerator = Accelerator(gradient_accumulation_steps=cfg["batches_per_step"])
    else:
        run = wandb.init(project="solu",
        entity="mechanistic-interpretability", 
        config=DEFAULT_CFG)
    
        print(run.config)
        cfg = dict(run.config)
        print("Setting run config to wandb")
        print(run.config["n_layers"])
        cfg = create_cfg(cfg)
        accelerator = Accelerator(gradient_accumulation_steps=cfg["batches_per_step"])

    set_seed(cfg["seed"])
    if accelerator.num_processes > 1:
        accelerator.print("Using accelerate!")
        cfg["use_acc"] = True 
    
    
    print("Is main process", accelerator.is_main_process)
    accelerator.print("initialized accelerator")
    accelerator.print(json.dumps(cfg, indent=2))

    if accelerator.is_main_process and MODE!="wandb":
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
        
    
    if accelerator.is_main_process and not cfg["debug"] and MODE!="wandb":
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
                    and MODE!="wandb"
                    and schedule.step()
                    and cfg["save_checkpoints"]
                    and not cfg["debug"]
                ):
                    accelerator.print(
                        f"Saved the model! Step: {step}. Frac of way through training: {schedule.schedule[schedule.next_save_point-1]}"
                    )
                    if not cfg["debug"]:
                        
                        torch.save(model.state_dict(), checkpoint_dir/f"tokens_{total_tokens:0>12}.pth")
                if accelerator.is_main_process and (step+1) % 500 == 0 and MODE!="wandb":
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

    wiki_loss = test_wikipedia(model, tokenizer).item()
    ind_loss = test_induction(model, tokenizer).item()

    wandb.log({"wiki_loss": wiki_loss, "ind_loss": ind_loss}, step=step)

    if not cfg["debug"] and accelerator.is_main_process and MODE != "wandb":
        torch.save(model.state_dict(), save_dir/f"model_final.pth")
        solu_utils.move_folder_to_hub(model_name, just_final=True)
        wandb.finish()
    
# %%
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