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


def create_cfg(accelerator):
    cfg = {
        "d_model": 512,
        "n_layers": 4,
        "lr": 1e-3,
        "batch_size": 36,
        "batches_per_step": 1,
        "seed": 98742,
        # 'checkpoint_every_tokens':5*10**7,
        "use_checkpoint_schedule": True,
        "debug": False,
        "debug_batch": False,
        "debug_overfit": False,
        "normalization": "LN",  # 'LN' 'RMS' or None
        # "max_tokens": 15 * 10 ** 9,
        "max_tokens": 22*10**9,
        "version": 36,
        "use_float16": False,
        "use_bfloat16": False,
        "save_checkpoints_to_bfloat16": True,
        "use_bfloat16_matmul": True,
        "right_multiply_matrices": True,
        # 'n_heads':8,
        "d_head": 64,
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
        "warmup_tokens": 2*10**8,
        "factored_embed": False,
        "train_loss_ewma_beta": 0.99,
        "shuffled_data": True,
        # 'W_O_init_scale':True,
    }
    # accelerator.print('Old')
    # accelerato(cfg)
    # print()
    cfg["n_heads"] = cfg["d_model"] // cfg["d_head"]
    cfg["d_mlp"] = 4 * cfg["d_model"]
    cfg["tokens_per_step"] = cfg["batch_size"] * cfg["n_ctx"] * cfg["batches_per_step"] * 8
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


# save_to_bfloat16(model, 'SoLU_3L_testing.pth')

# A helper class to get access to intermediate activations (inspired by Garcon)
# It's a dummy module that is the identity function by default
# I can wrap any intermediate activation in a HookPoint and get a convenient
# way to add PyTorch hooks
class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
        self.ctx = {}

        # A variable giving the hook's name (from the perspective of the root
        # module) - this is set by the root module at setup.
        self.name = None

    def add_hook(self, hook, dir="fwd"):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, hook=self)

        if dir == "fwd":
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == "bwd":
            handle = self.register_full_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir="fwd"):
        if (dir == "fwd") or (dir == "both"):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir == "bwd") or (dir == "both"):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction {dir}")

    def clear_context(self):
        del self.ctx
        self.ctx = {}

    def forward(self, x):
        return x

    def layer(self):
        # Returns the layer index if the name has the form 'blocks.{layer}.{...}'
        # Helper function that's mainly useful on EasyTransformer
        # If it doesn't have this form, raises an error -
        split_name = self.name.split(".")
        return int(split_name[1])


class HookedRootModule(nn.Module):
    # A class building on nn.Module to interface nicely with HookPoints
    # Allows you to name each hook, remove hooks, cache every activation/gradient, etc
    def __init__(self, *args):
        super().__init__()

    def setup_hooks(self):
        # Setup function - this needs to be run in __init__ AFTER defining all
        # layers
        # Add a parameter to each module giving its name
        # Build a dictionary mapping a module name to the module
        self.mod_dict = {}
        self.hook_dict = {}
        for name, module in self.named_modules():
            module.name = name
            self.mod_dict[name] = module
            if type(module) == HookPoint:
                self.hook_dict[name] = module

    def hook_points(self):
        return self.hook_dict.values()

    def remove_all_hook_fns(self, direction="both"):
        for hp in self.hook_points():
            hp.remove_hooks(direction)

    def clear_contexts(self):
        for hp in self.hook_points():
            hp.clear_context()

    def reset_hooks(self, clear_contexts=True, direction="both"):
        if clear_contexts:
            self.clear_contexts()
        self.remove_all_hook_fns(direction)

    def cache_all(self, cache, incl_bwd=False, device="cuda"):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, hook):
            cache[hook.name] = tensor.detach().to(device)

        def save_hook_back(tensor, hook):
            cache[hook.name + "_grad"] = tensor[0].detach().to(device)

        for hp in self.hook_points():
            hp.add_hook(save_hook, "fwd")
            if incl_bwd:
                hp.add_hook(save_hook_back, "bwd")

    def run_with_hooks(
        self,
        *args,
        fwd_hooks=[],
        bwd_hooks=[],
        reset_hooks_start=True,
        reset_hooks_end=True,
        clear_contexts=False,
    ):
        """
        fwd_hooks: A list of (name, hook), where name is either the name of
        a hook point or a Boolean function on hook names and hook is the
        function to add to that hook point, or the hook whose names evaluate
        to True respectively. Ditto bwd_hooks
        reset_hooks_start (bool): If True, all prior hooks are removed at the start
        reset_hooks_end (bool): If True, all hooks are removed at the end (ie,
        including those added in this run)
        clear_contexts (bool): If True, clears hook contexts whenever hooks are reset

        Note that if we want to use backward hooks, we need to set
        reset_hooks_end to be False, so the backward hooks are still there - this function only runs a forward pass.
        """
        if reset_hooks_start:
            self.reset_hooks(clear_contexts)
        for name, hook in fwd_hooks:
            if type(name) == str:
                self.mod_dict[name].add_hook(hook, dir="fwd")
            else:
                # Otherwise, name is a Boolean function on names
                for hook_name, hp in self.hook_dict.items():
                    if name(hook_name):
                        hp.add_hook(hook, dir="fwd")
        for name, hook in bwd_hooks:
            if type(name) == str:
                self.mod_dict[name].add_hook(hook, dir="fwd")
            else:
                # Otherwise, name is a Boolean function on names
                for hook_name, hp in self.hook_dict:
                    if name(hook_name):
                        hp.add_hook(hook, dir="bwd")
        out = self.forward(*args)
        if reset_hooks_end:
            if len(bwd_hooks) > 0:
                print(
                    "WARNING: Hooks were reset at the end of run_with_hooks while backward hooks were set."
                )
                print(
                    "This removes the backward hooks before a backward pass can occur"
                )
            self.reset_hooks(clear_contexts)
        return out


def loss_fn(logits, batch):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, batch[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def amp_einsum(einsum_str, mat1, mat2, use_bfloat16=True):
    # return torch.einsum(einsum_str, mat1, mat2)
    # return torch.einsum(einsum_str, mat1.to(torch.bfloat16), mat2.to(torch.bfloat16)).to(torch.float32)
    if use_bfloat16:
        return torch.einsum(
            einsum_str, mat1.to(torch.bfloat16), mat2.to(torch.bfloat16)
        )
    else:
        return torch.einsum(einsum_str, mat1, mat2)


# Define network architecture

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(self.cfg["d_vocab"], self.cfg["d_model"]))
        nn.init.kaiming_uniform_(self.W_E, a=np.sqrt(5), mode="fan_out")

    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        # return einops.rearrange(self.W_E[tokens, :], 'd_model batch pos -> batch pos d_model')
        return self.W_E[tokens, :]


# class FactoredEmbed(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.W_E = nn.Parameter(torch.empty(self.cfg['factor_size'], self.cfg['d_vocab']))
#         self.W_E_factor = nn.Parameter(torch.empty(self.cfg['d_model'], self.cfg['factor_size']))
#         nn.init.kaiming_uniform_(self.W_E, a=np.sqrt(5), mode='fan_out')
#         nn.init.kaiming_uniform_(self.W_E_factor, a=np.sqrt(5), mode='fan_out')

#     def forward(self, tokens):
#         # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
#         # B acts as a tensor of indices into the second dimension (so >=0 and <b)
#         return einops.rearrange(self.W_E[:, tokens], 'factor batch pos -> batch pos factor') @ self.W_E_factor.T


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty(self.cfg["d_model"], self.cfg["d_vocab"]))
        nn.init.kaiming_uniform_(self.W_U, a=np.sqrt(5), mode="fan_out")

    def forward(self, residual):
        return amp_einsum(
            "bpm,mv->bpv", residual, self.W_U, self.cfg["use_bfloat16_matmul"]
        )  # [batch, pos, d_vocab]


# class FactoredUnembed(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.W_U = nn.Parameter(torch.empty(self.cfg['d_vocab'], self.cfg['factor_size']))
#         self.W_U_factor = nn.Parameter(torch.empty(self.cfg['factor_size'], self.cfg['d_model']))
#         nn.init.kaiming_uniform_(self.W_U, a=np.sqrt(5), mode='fan_out')
#         nn.init.kaiming_uniform_(self.W_U_factor, a=np.sqrt(5), mode='fan_out')

#     def forward(self, residual):
#         # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
#         # B acts as a tensor of indices into the second dimension (so >=0 and <b)
#         return amp_einsum('fm,vf,bpm->bpv', self.W_U_factor, self.W_U, residual) # [batch, pos, d_vocab]

# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg["n_ctx"], self.cfg["d_model"]))
        nn.init.kaiming_uniform_(self.W_pos, a=np.sqrt(5), mode="fan_out")

    def forward(self, x):
        # Output shape [pos, d_model] - will be broadcast along batch dim
        return self.W_pos[: x.size(-1), :]  # [pos, d_model]


class LayerNormPre(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.eps = self.cfg["ln_eps"]

        # Adds a hook point for the normalization scale factor
        self.hook_scale = HookPoint()  # [batch, pos]

    def forward(self, x):
        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, d_model]
        scale = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )  # [batch, pos, 1]
        return x / scale


class LayerNorm(nn.Module):
    def __init__(self, cfg, length):
        super().__init__()
        self.cfg = cfg
        self.eps = self.cfg["ln_eps"]
        self.length = length
        self.w = nn.Parameter(torch.ones(length))
        self.b = nn.Parameter(torch.zeros(length))

        # Adds a hook point for the normalization scale factor
        self.hook_scale = HookPoint()  # [batch, pos]

    def forward(self, x):
        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, d_model]
        scale = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )  # [batch, pos, 1]
        out = (x / scale) * self.w + self.b
        return out


class RMSNorm(nn.Module):
    def __init__(self, cfg, length):
        super().__init__()
        self.cfg = cfg
        self.eps = self.cfg["ln_eps"]
        self.length = length
        self.w = nn.Parameter(torch.ones(length))

        # Adds a hook point for the normalization scale factor
        self.hook_scale = HookPoint()  # [batch, pos]

    def forward(self, x):
        scale = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )  # [batch, pos, 1]
        out = (x / scale) * self.w
        return out


# Attention
class Attention(nn.Module):
    def __init__(self, cfg, attn_type="global"):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(
            torch.empty(self.cfg["n_heads"], self.cfg["d_model"], self.cfg["d_head"])
        )
        self.b_Q = nn.Parameter(torch.zeros(self.cfg["n_heads"], self.cfg["d_head"]))
        nn.init.kaiming_uniform_(self.W_Q, a=np.sqrt(5), mode="fan_out")
        self.W_K = nn.Parameter(
            torch.empty(self.cfg["n_heads"], self.cfg["d_model"], self.cfg["d_head"])
        )
        self.b_K = nn.Parameter(torch.zeros(self.cfg["n_heads"], self.cfg["d_head"]))
        nn.init.kaiming_uniform_(self.W_K, a=np.sqrt(5), mode="fan_out")
        self.W_V = nn.Parameter(
            torch.empty(self.cfg["n_heads"], self.cfg["d_model"], self.cfg["d_head"])
        )
        self.b_V = nn.Parameter(torch.zeros(self.cfg["n_heads"], self.cfg["d_head"]))
        nn.init.kaiming_uniform_(self.W_V, a=np.sqrt(5), mode="fan_out")
        self.W_O = nn.Parameter(
            torch.empty(self.cfg["n_heads"], self.cfg["d_head"], self.cfg["d_model"])
        )
        self.b_O = nn.Parameter(torch.zeros(self.cfg["d_model"]))
        nn.init.kaiming_uniform_(self.W_O, a=np.sqrt(5), mode="fan_out")
        # if cfg['W_O_init_scale']:
        #     self.W_O/=np.sqrt(2*self.cfg['n_layers'])

        self.attn_type = attn_type
        # Create a query_pos x key_pos mask, with True iff that query position
        # can attend to that key position
        causal_mask = torch.tril(
            torch.ones((self.cfg["n_ctx"], self.cfg["n_ctx"])).bool()
        )
        self.register_buffer("mask", causal_mask)

        self.register_buffer("IGNORE", torch.tensor(-1e5))
        self.attn_scale = np.sqrt(self.cfg["d_head"])

        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_attn = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, head_index, head_index, d_model]
        if not cfg["use_pos_resid"]:
            self.hook_attn_input = HookPoint()

    def forward(self, x, pos_embed):
        if not self.cfg["use_pos_resid"]:
            attn_input = self.hook_attn_input(x + pos_embed)
            q = self.hook_q(
                amp_einsum(
                    "bpm,imh->bpih",
                    attn_input,
                    self.W_Q,
                    self.cfg["use_bfloat16_matmul"],
                )
                + self.b_Q
            )  # [batch, pos, head_index, d_head]
            k = self.hook_k(
                amp_einsum(
                    "bpm,imh->bpih",
                    attn_input,
                    self.W_K,
                    self.cfg["use_bfloat16_matmul"],
                )
                + self.b_K
            )  # [batch, pos, head_index, d_head]
        else:
            q = self.hook_q(
                amp_einsum(
                    "bpm,imh->bpih", x, self.W_Q, self.cfg["use_bfloat16_matmul"]
                )
                + self.b_Q
            )  # [batch, pos, head_index, d_head]
            k = self.hook_k(
                amp_einsum(
                    "bpm,imh->bpih", x, self.W_K, self.cfg["use_bfloat16_matmul"]
                )
                + self.b_K
            )  # [batch, pos, head_index, d_head]

        v = self.hook_v(
            amp_einsum("bpm,imh->bpih", x, self.W_V, self.cfg["use_bfloat16_matmul"])
            + self.b_V
        )  # [batch, pos, head_index, d_head]
        attn_scores = (
            amp_einsum(
                "bpih,bqih->bipq",
                q.to(torch.float32),
                k.to(torch.float32, self.cfg["use_bfloat16_matmul"]),
            )
            / self.attn_scale
        )  # [batch, head_index, query_pos, key_pos]
        attn_scores = self.hook_attn_scores(
            self.apply_causal_mask(attn_scores)
        )  # [batch, head_index, query_pos, key_pos]
        attn_matrix = self.hook_attn(
            F.softmax(attn_scores, dim=-1)
        )  # [batch, head_index, query_pos, key_pos]
        z = self.hook_z(
            amp_einsum(
                "bpih,biqp->bqih", v, attn_matrix, self.cfg["use_bfloat16_matmul"]
            )
        )  # [batch, pos, head_index, d_head]

        if self.cfg["use_attn_result"]:
            result = self.hook_result(
                amp_einsum(
                    "bqih,ihm->bqim", z, self.W_O, self.cfg["use_bfloat16_matmul"]
                )
            )  # [batch, pos, head_index, d_model]
            out = (
                einops.reduce(
                    result, "batch position index model->batch position model", "sum"
                )
                + self.b_O
            )  # [batch, pos, d_model]
        else:
            out = (
                amp_einsum(
                    "bqih,ihm->bqm", z, self.W_O, self.cfg["use_bfloat16_matmul"]
                )
                + self.b_O
            )  # [batch, pos, head_index, d_model]
        return out

    def apply_causal_mask(self, attn_scores):
        return torch.where(
            self.mask[: attn_scores.size(-2), : attn_scores.size(-1)],
            attn_scores,
            self.IGNORE,
        )


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(self.cfg["d_model"], self.cfg["d_mlp"]))
        nn.init.kaiming_uniform_(self.W_in, a=np.sqrt(5), mode="fan_out")
        self.b_in = nn.Parameter(torch.zeros(self.cfg["d_mlp"]))
        self.W_out = nn.Parameter(torch.empty(self.cfg["d_mlp"], self.cfg["d_model"]))
        nn.init.kaiming_uniform_(self.W_out, a=np.sqrt(5), mode="fan_out")
        self.b_out = nn.Parameter(torch.zeros(self.cfg["d_model"]))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg["act_fn"].lower() == "relu":
            self.act_fn = F.relu
        elif self.cfg["act_fn"].lower() == "gelu_new":
            self.act_fn = gelu_new
        elif self.cfg["act_fn"].lower() == "solu":
            self.act_fn = lambda x: F.softmax(x, dim=-1) * x
            self.hook_post_ln = HookPoint()  # [batch, pos, d_mlp]
            self.ln = LayerNorm(self.cfg, self.cfg["d_mlp"])
        else:
            raise ValueError(f"Invalid activation function name: {self.cfg['act_fn']}")

    def forward(self, x):
        x = self.hook_pre(
            amp_einsum("bpd,dm->bpm", x, self.W_in, self.cfg["use_bfloat16_matmul"])
            + self.b_in
        )  # [batch, pos, d_mlp]
        x = self.hook_post(self.act_fn(x))  # [batch, pos, d_mlp]
        if self.cfg["act_fn"].lower() == "solu":
            x = self.hook_post_ln(self.ln(x))
        x = (
            amp_einsum("bpm,md->bpd", x, self.W_out, self.cfg["use_bfloat16_matmul"])
            + self.b_out
        )  # [batch, pos, d_model]
        return x


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg, block_index):
        super().__init__()
        self.cfg = cfg
        if self.cfg["normalization"] == "RMS":
            self.norm1 = LayerNorm(self.cfg, self.cfg["d_model"])
            self.norm2 = LayerNorm(self.cfg, self.cfg["d_model"])
        elif self.cfg["normalization"] == "LN":
            self.norm1 = LayerNorm(self.cfg, self.cfg["d_model"])
            self.norm2 = LayerNorm(self.cfg, self.cfg["d_model"])
        self.attn = Attention(self.cfg)
        self.mlp = MLP(self.cfg)

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        # Note that resid_pre of layer k+1 is resid_post of layer k - given for convenience
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(self, x, pos_embed):
        resid_pre = self.hook_resid_pre(x)  # [batch, pos, d_model]
        if self.cfg["normalization"] is not None:
            attn_out = self.hook_attn_out(
                self.attn(self.norm1(resid_pre), pos_embed)
            )  # [batch, pos, d_model]
        else:
            attn_out = self.hook_attn_out(
                self.attn(resid_pre, pos_embed)
            )  # [batch, pos, d_model]
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)  # [batch, pos, d_model]
        if self.cfg["normalization"] is not None:
            mlp_out = self.hook_mlp_out(
                self.mlp(self.norm2(resid_mid))
            )  # [batch, pos, d_model]
        else:
            mlp_out = self.hook_mlp_out(self.mlp(resid_mid))  # [batch, pos, d_model]
        resid_post = self.hook_resid_post(resid_mid + mlp_out)  # [batch, pos, d_model]
        return resid_post


# Full transformer
class Transformer(HookedRootModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

        if self.cfg["factored_embed"]:
            self.embed = FactoredEmbed(self.cfg)
        else:
            self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint()  # [batch, pos, d_model]

        if cfg["normalization"] == "RMS":
            self.norm = RMSNorm(self.cfg, self.cfg["d_model"])
        elif cfg["normalization"] == "LN":
            self.norm = LayerNorm(self.cfg, self.cfg["d_model"])

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.cfg, block_index)
                for block_index in range(self.cfg["n_layers"])
            ]
        )

        if self.cfg["factored_embed"]:
            self.unembed = FactoredUnembed(self.cfg)
        else:
            self.unembed = Unembed(self.cfg)

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup_hooks()

    def forward(self, tokens, return_loss=True):
        # Input x is either a batch of tokens ([batch, pos]) or a text string
        # if type(x)==str:
        #     # If text, convert to tokens (batch_size=1)
        #     x = self.to_tokens(x)
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(self.pos_embed(tokens))  # [batch, pos, d_model]
        if self.cfg["use_pos_resid"]:
            residual = embed + pos_embed  # [batch, pos, d_model]
        else:
            residual = embed  # [batch, pos, d_model]
        for block in self.blocks:
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(residual, pos_embed)  # [batch, pos, d_model]
        if self.cfg["normalization"] is not None:
            residual = self.norm(residual)
        logits = self.unembed(residual.to(torch.float32))  # [batch, pos, d_vocab]
        if return_loss:
            return loss_fn(logits, tokens)
        else:
            return logits

    def to_tokens(self, text):
        return self.tokenizer(text, return_tensors="pt")["input_ids"]


def init_tokenizer(accelerator):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    pad_token = "<PAD>"
    tokenizer.add_special_tokens({"pad_token": pad_token})
    accelerator.print(tokenizer)
    return tokenizer


def create_dataset(cfg, accelerator):
    tokenizer = init_tokenizer(accelerator)
    seq_len = cfg["n_ctx"]

    def tokenize(examples):
        start_time = time.time()
        texts = examples["text"]
        full_text = tokenizer.eos_token.join(texts)
        div = 20
        length = len(full_text) // div
        text_list = [full_text[i * length : (i + 1) * length] for i in range(div)]
        tokens = tokenizer(text_list, return_tensors="np", padding=True)[
            "input_ids"
        ].flatten()
        tokens = tokens[tokens != tokenizer.pad_token_id]
        # print(len(text_list), len(text_list[0]))
        # print(tokens.shape)
        n = len(tokens)
        curr_batch_size = n // (seq_len - 1)
        tokens = tokens[: (seq_len - 1) * curr_batch_size]
        tokens = einops.rearrange(
            tokens,
            "(batch_size seq) -> batch_size seq",
            batch_size=curr_batch_size,
            seq=seq_len - 1,
        )
        prefix = np.ones((curr_batch_size, 1), dtype=np.int64) * tokenizer.bos_token_id
        # print(tokens.shape, n, curr_batch_size, seq_len)
        return {
            "text": np.concatenate([prefix, tokens], axis=1)
        }  # tiny_owt_orig_2 = load_dataset('stas/openwebtext-10k', cache_dir='./cache', split='train', download_config=datasets.DownloadConfig(resume_download=True, num_proc=4))

    import time

    if not cfg["debug"]:
        start_time = time.time()
        if cfg["dataset_name"] == "the_pile":
            if cfg["shuffled_data"]:
                randperm = np.random.permutation(28)
                accelerator.print("Permutation of PILE URLs", randperm)
                pile_urls = [
                    f"https://the-eye.eu/public/AI/pile/train/{i:0>2}.jsonl.zst"
                    for i in randperm
                ]
                accelerator.print("Pile URLs: ", pile_urls)
                dataset = load_dataset(
                    "json", data_files=pile_urls, streaming=True, split="train"
                )
            else:
                dataset = load_dataset(cfg["dataset_name"], streaming=True, split="train")
            accelerator.print("Loaded!", time.time() - start_time)
            start_time = time.time()
            try:
                dataset = dataset.remove_columns("meta")
            except:
                accelerator.print("Meta not in dataset")

            accelerator.print("Loaded!", time.time() - start_time)
            start_time = time.time()
            dataset = dataset.map(tokenize, batched=True)
            accelerator.print("dataset.map", time.time() - start_time)
            start_time = time.time()
            dataset = dataset.with_format(type="torch")
            accelerator.print("dataset.set_format", time.time() - start_time)
            start_time = time.time()
            dataset = dataset.shuffle(seed=cfg["seed"], buffer_size=30000)
            accelerator.print("dataset.shuffle", time.time() - start_time)
            start_time = time.time()
            train_data_loader = DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=3)
            accelerator.print("train_data_loader =", time.time() - start_time)
        elif cfg["dataset_name"] == "wikipedia":
            dataset = load_dataset(cfg["dataset_name"], cfg["dataset_subset_name"], split="train", cache_dir="cache")
            try:
                dataset = dataset.remove_columns(["id", "url", "title"])
            except:
                accelerator.print("Id url or title not in dataset")
            accelerator.print("Loaded!", time.time() - start_time)
            start_time = time.time()
            dataset = dataset.map(tokenize, batched=True, num_proc=16)
            accelerator.print("dataset.map", time.time() - start_time)
            start_time = time.time()
            dataset = dataset.with_format(type="torch")
            accelerator.print("dataset.set_format", time.time() - start_time)
            start_time = time.time()
            dataset = dataset.shuffle(seed=cfg["seed"])
            train_data_loader = DataLoader(dataset, batch_size=cfg["batch_size"])
            accelerator.print("train_data_loader =", time.time() - start_time)
        elif cfg["dataset_name"] == "c4":
            dataset = load_dataset('c4', "en", streaming=True, split='train')
            dataset = dataset.remove_columns(['url', 'timestamp'])
            dataset = dataset.map(tokenize, batched=True)
            dataset = dataset.with_format(type='torch')
            dataset = dataset.shuffle(seed=cfg['seed'])
            train_data_loader = DataLoader(dataset, batch_size=cfg['batch_size'])
        else:
            raise ValueError(f"Invalid Dataset Name: {cfg['dataset_name']}")

    else:
        if cfg['dataset_name']=='c4':
            dataset = load_dataset('c4', "en", streaming=True, split='train')
            dataset = dataset.remove_columns(['url', 'timestamp'])
            dataset = dataset.map(tokenize, batched=True)
            dataset = dataset.with_format(type='torch')
            dataset = dataset.shuffle(seed=cfg['seed'])
            train_data_loader = DataLoader(dataset, batch_size=cfg['batch_size'])
        else:
            streaming_owt = load_dataset(
                "stas/openwebtext-10k", split="train", cache_dir="cache"
            )
            streaming_owt = streaming_owt.map(tokenize, batched=True, num_proc=10)
            streaming_owt = streaming_owt.with_format(type="torch")
            train_data_loader = DataLoader(
                streaming_owt, batch_size=cfg["batch_size"], shuffle=True
            )
            start_time = time.time()
            for c, i in tqdm.tqdm(enumerate(train_data_loader)):
                if c == 0:
                    accelerator.print("Loaded Initial stream!")
                    accelerator.print(c, time.time() - start_time)
                    start_time = time.time()
                elif c == 1:
                    accelerator.print("Time for next batch:", time.time() - start_time)
                    break
    return train_data_loader

class SaveSchedule:
    def __init__(self, max_tokens, tokens_per_step, schedule=None):
        if schedule is None:
            self.schedule = np.concatenate(
                [
                    np.arange(10) / 10 * 1e-3,
                    np.arange(2, 20) / 20 * 1e-2,
                    np.arange(5, 50) / 50 * 1e-1,
                    np.arange(10, 101) / 100,
                ]
            )
        else:
            self.schedule = schedule
        self.max_tokens = max_tokens
        self.tokens_per_step = tokens_per_step
        self.counter = 0
        self.next_save_point = 0
        px.line(
            self.schedule * max_tokens,
            log_y=True,
            title="Save Schedule",
            labels={"y": "Tokens", "x": "Checkpoint Index"},
        ).show()

    def step(self):
        value = self.counter * self.tokens_per_step / self.max_tokens
        threshold = self.schedule[self.next_save_point]
        if value >= threshold:
            self.next_save_point += 1
            self.counter += 1
            return True
        else:
            self.counter += 1
            return False


def main(mixed_precision="bf16", seed: int = 42):
    set_seed(seed)

    accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=1)

    cfg = create_cfg(accelerator)
    assert cfg["batches_per_step"] == accelerator.gradient_accumulation_steps

    data_iter = create_dataset(cfg, accelerator)

    accelerator.print("initialized accelerator")

    model_name = f'SoLU_{cfg["n_layers"]}L_v{cfg["version"]}'

    if accelerator.is_main_process:
        wandb.init(project="solu", entity="mechanistic-interpretability", config=cfg)

        torch.save(cfg, model_name + "_config.pth")
        wandb.save(model_name + "_config.pth")

    tokenizer = init_tokenizer(accelerator)
    # device = accelerator.device

    if cfg["attn_only"]:
        model = AttnOnlyTransformer(cfg, tokenizer)
    else:
        model = Transformer(cfg, tokenizer)

    model.to(accelerator.device)
    if cfg["use_bfloat16"]:
        model.to(torch.bfloat16)
    elif cfg["use_float16"]:
        model.to(torch.float16)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=cfg["betas"],
        weight_decay=cfg["weight_decay"],
    )
    if cfg["lr_schedule"] is not None:

        def lr_schedule(step):
            if step < cfg["warmup_steps"]:
                return (1e-7 + (cfg["lr"] - 1e-7) * step / cfg["warmup_steps"]) / cfg[
                    "lr"
                ]
            else:
                return 0.55 + 0.9 * 0.5 * np.cos(
                    np.pi
                    * (step - cfg["warmup_steps"])
                    / (cfg["max_steps"] - cfg["warmup_steps"])
                )

        param_groups = {"decay": [], "no_decay": []}
        for name, param in model.named_parameters():
            accelerator.print(name)
            accelerator.print(param.dtype)
            if "W_" in name and name not in ["W_E", "W_U"]:
                param_groups["decay"].append(param)
            else:
                param_groups["no_decay"].append(param)
        optim_groups = [
            {"params": param_groups["decay"], "weight_decay": cfg["weight_decay"]},
            {"params": param_groups["no_decay"], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=cfg["lr"])
        accelerator.print(optimizer)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    if accelerator.is_main_process:
        schedule = SaveSchedule(
            cfg["max_tokens"],
            cfg["tokens_per_step"],
        )

    model, optimizer, data_iter, scheduler = accelerator.prepare(model, optimizer, data_iter, scheduler)

    accelerator.print(cfg)
    # DataLoader(full_owt_test['text'], batch_size=cfg['batch_size'], shuffle=False, pin_memory=False)
    accelerator.print("Training begins!")
    losses = []
    loss_ewmas = []
    step = 0
    start_time = time.time()
    loss_ewma = 9
    # loss_beta = 0.95
    total_tokens = 0
    running_loss = 0
    prev_time = time.time()
    epoch = 0
    # for epoch in range(100):
    for c, batch in tqdm.tqdm(enumerate(data_iter)):
        with accelerator.accumulate(model):
            batch = batch["text"]
            if cfg["debug"] and epoch == 0 and c < 3 and accelerator.is_main_process:
                accelerator.print(batch[0])
                accelerator.print(tokenizer.decode(batch[0]))
            # batch = batch.cuda()
            loss = model(batch)
            # loss = loss / accelerator.num_processes
            accelerator.backward(loss)

            # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            running_loss += accelerator.reduce(loss.detach(), "mean").item() * accelerator.gradient_accumulation_steps
            batch_tokens = torch.tensor(batch.numel(), device=accelerator.device)*8
            # dist.all_reduce(batch_tokens, op=dist.ReduceOp.SUM)
            
            total_tokens += batch_tokens.item()
            if (c + 1) % cfg["batches_per_step"] == 0:
                # accelerator.clip_grad_norm_(model.parameters(), cfg["grad_norm_clip"])
                optimizer.step()
                if cfg["lr_schedule"] is not None:
                    scheduler.step()
                    if accelerator.is_main_process:
                        wandb.log({"scheduled_lr": scheduler.get_last_lr()[0]}, step=step)
                optimizer.zero_grad()
                if (
                    accelerator.is_main_process
                    and schedule.step()
                    and cfg["use_checkpoint_schedule"]
                ):
                    accelerator.print(
                        f"Saved the model! Step: {step}. Frac of way through training: {schedule.schedule[schedule.next_save_point-1]*accelerator.num_processes}"
                    )
                    if not cfg["debug"]:
                        if cfg["save_checkpoints_to_bfloat16"]:
                            save_to_bfloat16(model, f"{model_name}_{step:0>6}.pth")
                        else:
                            torch.save(model.state_dict(), f"{model_name}_{step:0>6}.pth")
                        torch.save(
                            optimizer.state_dict(), f"{model_name}_opt_checkpoint.pth"
                        )
                        if cfg["lr_schedule"] is not None:
                            torch.save(
                                scheduler.state_dict(),
                                f"{model_name}_scheduler_checkpoint.pth",
                            )
                        wandb.save(f"{model_name}_{step:0>6}.pth")
                running_loss = running_loss / cfg["batches_per_step"]
                losses.append(running_loss)

                loss_ewma = loss_ewma * cfg["train_loss_ewma_beta"] + running_loss * (
                    1 - cfg["train_loss_ewma_beta"]
                )
                loss_ewmas.append(loss_ewma)
                if accelerator.is_main_process:
                    wandb.log(
                        {
                            "loss": loss.item() * accelerator.gradient_accumulation_steps,
                            "loss_ewma": loss_ewma,
                            "elapsed": time.time() - start_time,
                            "total_tokens": total_tokens,
                            "c": c,
                        },
                        step=step,
                    )
                # accelerator.print("Just logged")
                # accelerator.print(
                #     {
                #         "loss": loss.item(),
                #         "loss_ewma": loss_ewma,
                #         "elapsed": time.time() - start_time,
                #         "total_tokens": total_tokens,
                #         "c": c,
                #     }
                # )
                running_loss = 0
                if step % 30 == 0:
                    accelerator.print(c, step, total_tokens, losses[-1], loss_ewmas[-1])
                step += 1
                # if step >= cfg["max_steps"]:
                #     break
                if total_tokens > cfg["max_tokens"]:
                    break
            if c <= 12 and epoch == 0:
                cuda_memory()
                accelerator.print("Early iteration complete!", c, time.time() - prev_time)
                prev_time = time.time()
            del loss
        # print(batch.shape, logits.shape, running_loss, loss, step, total_tokens)
        # if not cfg['debug_overfit']:
        #     break

    accelerator.print(f"Finished training! Train Loss EWMA: {loss_ewma}")

    if not cfg["debug"] and accelerator.is_main_process:
        torch.save(model.state_dict(), f"{model_name}_final.pth")
        wandb.save(f"{model_name}_final.pth")
        wandb.finish()


if __name__ == "__main__":
    main()
