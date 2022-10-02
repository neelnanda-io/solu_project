# %%
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops

import tqdm.notebook as tqdm

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

from functools import partial
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
import sys
from pprint import pprint
import math

import pysvelte
from easy_transformer import EasyTransformer, HookedRootModule, HookPoint
from rich import print


ROOT = Path('/workspace/solu_project/')
DTYPE = torch.bfloat16
test_var = 1
# %%
import tqdm.auto as tqdm
for i in tqdm.tqdm(range(1)):
    print("^ TQDM Should be working")
# %%

if "ipykernel_launcher" in os.path.basename(sys.argv[0]):
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic("matplotlib inline")
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    print("Activated reload")
# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
dataset = datasets.load_from_disk(ROOT/"pile_29.hf")
dataset = dataset.with_format(type='torch')
# %%
# import time
# print("Sleeping")
# time.sleep(10**2)
# print("Slept")
# f = open("test.txt", "w")
# f.write("Hello")
# f.close()
def cuda_memory():
    print(torch.cuda.memory_allocated()/1024**3, torch.cuda.memory_cached()/1024**3)


# %%
cfg = {
    'd_model':736,
    'normalization':'RMS', # 'LN' 'RMS' or None
    'model_checkpoint_name':'SoLU_2L_v10_final.pth',
    'n_layers':2,
    # 'd_model':1024,
    # 'normalization':'RMS', # 'LN' 'RMS' or None
    # 'model_checkpoint_name':'SoLU_1L_v9_final.pth',
    # 'n_layers':1,
    # 'd_model':128,
    # 'normalization':None, # 'LN' 'RMS' or None
    # 'model_checkpoint_name':'SoLU_1L_v11_final.pth',
    # 'n_layers':1,
    # 'n_heads':8,
    'd_head':64,
    'n_ctx':1024,
    'd_vocab':50278,
    # 'factor_size':256,
    'lr':1e-3,
    'betas':(0.9, 0.99),
    'weight_decay':0.01,
    'batch_size':40 * torch.cuda.device_count(),
    'batches_per_step':1,
    'seed':5000,
    'dataset_name':'the_pile',
    'grad_norm_clip':1.0,
    'checkpoint_every_tokens':1*10**8,
    'use_attn_result':False,
    'debug':False,
    'debug_batch':False,
    'debug_overfit':False,
    'max_tokens':15*10**9,
    'n_devices':torch.cuda.device_count(),
    'act_fn':'SoLU',
    'use_pos_resid':True,
    'attn_only':False,
    'ln_eps':1e-5,
    'version':10,
    'lr_schedule': 'cosine_warmup',
    'warmup_tokens':25*10**7,
    'factored_embed':False,
    'train_loss_ewma_beta':0.99,
    'device':'cuda',
    # 'W_O_init_scale':True,
}
print('Old')
pprint(cfg)
print()
cfg['n_heads'] = cfg['d_model']//cfg['d_head']
cfg['d_mlp'] = 4 * cfg['d_model']
cfg['tokens_per_step'] = (cfg['batch_size']*cfg['n_ctx']*cfg['batches_per_step'])
cfg['max_steps'] = cfg['max_tokens']//cfg['tokens_per_step']
cfg['warmup_steps'] = cfg['warmup_tokens']//cfg['tokens_per_step']
cfg['checkpoint_every'] = cfg['checkpoint_every_tokens']//cfg['tokens_per_step']
if cfg['debug'] and not cfg['debug_overfit']:
    print('Old max steps:', cfg['max_steps'])
    cfg['max_steps']=20
# cfg['warmup_steps']=cfg['warmup_tokens']//cfg['tokens_per_step']
pprint(cfg)
torch.manual_seed(cfg['seed'])
np.random.seed(cfg['seed'])
random.seed(cfg['seed'])
# %%
#Plotting functions
# This is mostly a bunch of over-engineered mess to hack Plotly into producing 
# the pretty pictures I want, I recommend not reading too closely unless you 
# want Plotly hacking practice
def to_numpy(tensor, flat=False):
    if type(tensor)!=torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()
def imshow(tensor, xaxis=None, yaxis=None, animation_name='Snapshot', **kwargs):
    if tensor.shape[0]==p*p:
        tensor = unflatten_first(tensor)
    tensor = torch.squeeze(tensor)
    px.imshow(to_numpy(tensor, flat=False), 
              labels={'x':xaxis, 'y':yaxis, 'animation_name':animation_name}, 
              **kwargs).show()
# Set default colour scheme
imshow_pos = partial(imshow, color_continuous_scale='Blues')
# Creates good defaults for showing divergent colour scales (ie with both 
# positive and negative values, where 0 is white)
imshow = partial(imshow, color_continuous_scale='RdBu', color_continuous_midpoint=0.0)
# Presets a bunch of defaults to imshow to make it suitable for showing heatmaps 
# of activations with x axis being input 1 and y axis being input 2.
inputs_heatmap = partial(imshow, xaxis='Input 1', yaxis='Input 2', color_continuous_scale='RdBu', color_continuous_midpoint=0.0)
def line(x, y=None, hover=None, xaxis='', yaxis='', **kwargs):
    if type(y)==torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x)==torch.Tensor:
        x=to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()
def scatter(x, y, **kwargs):
    px.scatter(x=to_numpy(x, flat=True), y=to_numpy(y, flat=True), **kwargs).show()
def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()
def line_marker(x, **kwargs):
    lines([x], mode='lines+markers', **kwargs)
def animate_lines(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, xaxis='x', yaxis='y', **kwargs):
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[1]):
            rows.append([lines_list[i][j], snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=[yaxis, snapshot, xaxis])
    px.line(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover,**kwargs).show()

def animate_multi_lines(lines_list, y_index=None, snapshot_index = None, snapshot='snapshot', hover=None, swap_y_animate=False, **kwargs):
    # Can plot an animation of lines with multiple lines on the plot.
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if swap_y_animate:
        lines_list = lines_list.transpose(1, 0, 2)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if y_index is None:
        y_index = [str(i) for i in range(lines_list.shape[1])]
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append(list(lines_list[i, :, j])+[snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=y_index+[snapshot, 'x'])
    px.line(df, x='x', y=y_index, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover, **kwargs).show()

def animate_scatter(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, yaxis='y', xaxis='x', color=None, color_name = 'color', **kwargs):
    # Can plot an animated scatter plot
    # lines_list has shape snapshot x 2 x line
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    if color is None:
        color = np.ones(lines_list.shape[-1])
    if type(color)==torch.Tensor:
        color = to_numpy(color)
    if len(color.shape)==1:
        color = einops.repeat(color, 'x -> snapshot x', snapshot=lines_list.shape[0])
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append([lines_list[i, 0, j].item(), lines_list[i, 1, j].item(), snapshot_index[i], color[i, j]])
    print([lines_list[:, 0].min(), lines_list[:, 0].max()])
    print([lines_list[:, 1].min(), lines_list[:, 1].max()])
    df = pd.DataFrame(rows, columns=[xaxis, yaxis, snapshot, color_name])
    px.scatter(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_x=[lines_list[:, 0].min(), lines_list[:, 0].max()], range_y=[lines_list[:, 1].min(), lines_list[:, 1].max()], hover_name=hover, color=color_name, **kwargs).show()
line(np.arange(5))
# %%
def loss_fn(logits, batch):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, batch[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()

def per_token_loss_fn(logits, batch):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, batch[:, 1:, None])[..., 0]
    return -pred_log_probs
# %%

# Define network architecture

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(self.cfg['d_model'], self.cfg['d_vocab']))
        nn.init.kaiming_uniform_(self.W_E, a=np.sqrt(5))
    
    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return einops.rearrange(self.W_E[:, tokens], 'd_model batch pos -> batch pos d_model')

class FactoredEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(self.cfg['factor_size'], self.cfg['d_vocab']))
        self.W_E_factor = nn.Parameter(torch.empty(self.cfg['d_model'], self.cfg['factor_size']))
        nn.init.kaiming_uniform_(self.W_E, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_E_factor, a=np.sqrt(5))
    
    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return einops.rearrange(self.W_E[:, tokens], 'factor batch pos -> batch pos factor') @ self.W_E_factor.T


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty(self.cfg['d_vocab'], self.cfg['d_model']))
        nn.init.kaiming_uniform_(self.W_U, a=np.sqrt(5))
    
    def forward(self, tokens):
        return torch.einsum('vm,bpm->bpv', self.W_U, tokens) # [batch, pos, d_vocab]

class FactoredUnembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty(self.cfg['d_vocab'], self.cfg['factor_size']))
        self.W_U_factor = nn.Parameter(torch.empty(self.cfg['factor_size'], self.cfg['d_model']))
        nn.init.kaiming_uniform_(self.W_U, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_U_factor, a=np.sqrt(5))
    
    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return torch.einsum('fm,vf,bpm->bpv', self.W_U_factor, self.W_U, tokens) # [batch, pos, d_vocab]

# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg['d_model'], self.cfg['n_ctx'])) 
        nn.init.kaiming_uniform_(self.W_pos, a=np.sqrt(5))
    
    def forward(self, x):
        # Output shape [pos, d_model] - will be broadcast along batch dim
        return self.W_pos[:, :x.size(-1)].T # [pos, d_model]

class LayerNormPre(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.eps = self.cfg['ln_eps']

        # Adds a hook point for the normalization scale factor
        self.hook_scale = HookPoint() # [batch, pos]
    
    def forward(self, x):
        x = x - x.mean(axis=-1, keepdim=True) # [batch, pos, d_model]
        scale = self.hook_scale(x.pow(2).mean(-1, keepdim=True) + 
                                 self.eps).sqrt() # [batch, pos, 1]
        return x / scale

class LayerNorm(nn.Module):
    def __init__(self, cfg, length):
        super().__init__()
        self.cfg = cfg
        self.eps = self.cfg['ln_eps']
        self.length = length
        self.w = nn.Parameter(torch.ones(length))
        self.b = nn.Parameter(torch.zeros(length))

        # Adds a hook point for the normalization scale factor
        self.hook_scale = HookPoint() # [batch, pos]
    
    def forward(self, x):
        x = x - x.mean(axis=-1, keepdim=True) # [batch, pos, d_model]
        scale = self.hook_scale(x.pow(2).mean(-1, keepdim=True) + 
                                 self.eps).sqrt() # [batch, pos, 1]
        out = (x / scale) * self.w + self.b
        return out

class RMSNorm(nn.Module):
    def __init__(self, cfg, length):
        super().__init__()
        self.cfg = cfg
        self.eps = self.cfg['ln_eps']
        self.length = length
        self.w = nn.Parameter(torch.ones(length))

        # Adds a hook point for the normalization scale factor
        self.hook_scale = HookPoint() # [batch, pos]
    
    def forward(self, x):
        scale = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + 
                                 self.eps).sqrt()) # [batch, pos, 1]
        out = (x / scale) * self.w
        return out

# Attention
class Attention(nn.Module):
    def __init__(self, cfg, attn_type='global'):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty(self.cfg['n_heads'], self.cfg['d_head'], self.cfg['d_model']))
        self.b_Q = nn.Parameter(torch.zeros(self.cfg['n_heads'], self.cfg['d_head']))
        nn.init.kaiming_uniform_(self.W_Q, a=np.sqrt(5))
        self.W_K = nn.Parameter(torch.empty(self.cfg['n_heads'], self.cfg['d_head'], self.cfg['d_model']))
        self.b_K = nn.Parameter(torch.zeros(self.cfg['n_heads'], self.cfg['d_head']))
        nn.init.kaiming_uniform_(self.W_K, a=np.sqrt(5))
        self.W_V = nn.Parameter(torch.empty(self.cfg['n_heads'], self.cfg['d_head'], self.cfg['d_model']))
        self.b_V = nn.Parameter(torch.zeros(self.cfg['n_heads'], self.cfg['d_head']))
        nn.init.kaiming_uniform_(self.W_V, a=np.sqrt(5))
        self.W_O = nn.Parameter(torch.empty(self.cfg['n_heads'], self.cfg['d_model'], self.cfg['d_head']))
        self.b_O = nn.Parameter(torch.zeros(self.cfg['d_model']))
        nn.init.kaiming_uniform_(self.W_O, a=np.sqrt(5))
        # if cfg['W_O_init_scale']:
        #     self.W_O/=np.sqrt(2*self.cfg['n_layers'])
        
        self.attn_type = attn_type
        # Create a query_pos x key_pos mask, with True iff that query position 
        # can attend to that key position
        causal_mask = torch.tril(torch.ones((self.cfg['n_ctx'], self.cfg['n_ctx'])).bool())
        self.register_buffer('mask', causal_mask)
        
        self.register_buffer('IGNORE', torch.tensor(-1e5))
        self.attn_scale = np.sqrt(self.cfg['d_head'])
        
        self.hook_k = HookPoint() # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint() # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint() # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint() # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint() # [batch, head_index, query_pos, key_pos]
        self.hook_attn = HookPoint() # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint() # [batch, head_index, head_index, d_model]
        if not cfg['use_pos_resid']:
            self.hook_attn_input = HookPoint()

    def forward(self, x, pos_embed):
        if not cfg['use_pos_resid']:
            attn_input = self.hook_attn_input(x+pos_embed)
            q = self.hook_q(torch.einsum('ihm,bpm->bpih', self.W_Q, attn_input)+self.b_Q) # [batch, pos, head_index, d_head]
            k = self.hook_k(torch.einsum('ihm,bpm->bpih', self.W_K, attn_input)+self.b_K) # [batch, pos, head_index, d_head]
        else:
            q = self.hook_q(torch.einsum('ihm,bpm->bpih', self.W_Q, x)+self.b_Q) # [batch, pos, head_index, d_head]
            k = self.hook_k(torch.einsum('ihm,bpm->bpih', self.W_K, x)+self.b_K) # [batch, pos, head_index, d_head]

        v = self.hook_v(torch.einsum('ihm,bpm->bpih', self.W_V, x)+self.b_V) # [batch, pos, head_index, d_head]
        attn_scores = torch.einsum('bpih,bqih->bipq', q, k)/self.attn_scale # [batch, head_index, query_pos, key_pos]
        attn_scores = self.hook_attn_scores(self.apply_causal_mask(attn_scores)) # [batch, head_index, query_pos, key_pos]
        attn_matrix = self.hook_attn(F.softmax(attn_scores, dim=-1)) # [batch, head_index, query_pos, key_pos]
        z = self.hook_z(torch.einsum('bpih,biqp->bqih', v, attn_matrix)) # [batch, pos, head_index, d_head]
        
        if cfg['use_attn_result']:
            result = self.hook_result(torch.einsum('imh,bqih->bqim', self.W_O, z)) # [batch, pos, head_index, d_model]
            out = einops.reduce(result, 
                            'batch position index model->batch position model', 
                            'sum')+self.b_O  # [batch, pos, d_model]
        else:
            out = (torch.einsum('imh,bqih->bqm', self.W_O, z)+self.b_O) # [batch, pos, head_index, d_model]
        return out
    
    def apply_causal_mask(self, attn_scores):
        return torch.where(self.mask[:attn_scores.size(-2), :attn_scores.size(-1)], attn_scores, self.IGNORE)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(self.cfg['d_mlp'], self.cfg['d_model']))
        nn.init.kaiming_uniform_(self.W_in, a=np.sqrt(5))
        self.b_in = nn.Parameter(torch.zeros(self.cfg['d_mlp']))
        self.W_out = nn.Parameter(torch.empty(self.cfg['d_model'], self.cfg['d_mlp']))
        nn.init.kaiming_uniform_(self.W_out, a=np.sqrt(5))
        self.b_out = nn.Parameter(torch.zeros(self.cfg['d_model']))

        self.hook_pre = HookPoint() # [batch, pos, d_mlp]
        self.hook_post = HookPoint() # [batch, pos, d_mlp]

        if self.cfg['act_fn'].lower()=='relu':
            self.act_fn = F.relu
        elif self.cfg['act_fn'].lower()=='gelu_new':
            self.act_fn = gelu_new
        elif self.cfg['act_fn'].lower()=='solu':
            self.act_fn = lambda x: F.softmax(x, dim=-1)*x
            self.hook_post_ln = HookPoint() # [batch, pos, d_mlp]
            self.ln = LayerNorm(self.cfg, self.cfg['d_mlp'])
        else:
            raise ValueError(f"Invalid activation function name: {self.cfg['act_fn']}")

    def forward(self, x):
        x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in) # [batch, pos, d_mlp]
        x = self.hook_post(self.act_fn(x)) # [batch, pos, d_mlp]
        if self.cfg['act_fn'].lower()=='solu':
            x = self.hook_post_ln(self.ln(x))
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out # [batch, pos, d_model]
        return x

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg, block_index):
        super().__init__()
        self.cfg = cfg
        if self.cfg['normalization']=='RMS':
            self.norm1 = LayerNorm(self.cfg, self.cfg['d_model'])
            self.norm2 = LayerNorm(self.cfg, self.cfg['d_model'])
        elif self.cfg['normalization']=='LN':
            self.norm1 = LayerNorm(self.cfg, self.cfg['d_model'])
            self.norm2 = LayerNorm(self.cfg, self.cfg['d_model'])
        self.attn = Attention(self.cfg)
        self.mlp = MLP(self.cfg)

        self.hook_attn_out = HookPoint() # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint() # [batch, pos, d_model]
        # Note that resid_pre of layer k+1 is resid_post of layer k - given for convenience
        self.hook_resid_pre = HookPoint() # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint() # [batch, pos, d_model]
        self.hook_resid_post = HookPoint() # [batch, pos, d_model]
    
    def forward(self, x, pos_embed):
        resid_pre = self.hook_resid_pre(x) # [batch, pos, d_model]
        if self.cfg['normalization'] is not None:
            attn_out = self.hook_attn_out(self.attn(self.norm1(resid_pre), pos_embed)) # [batch, pos, d_model]
        else:
            attn_out = self.hook_attn_out(self.attn(resid_pre, pos_embed)) # [batch, pos, d_model]
        resid_mid = self.hook_resid_mid(resid_pre + attn_out) # [batch, pos, d_model]
        if self.cfg['normalization'] is not None:
            mlp_out = self.hook_mlp_out(self.mlp(self.norm2(resid_mid))) # [batch, pos, d_model]
        else:
            mlp_out = self.hook_mlp_out(self.mlp(resid_mid)) # [batch, pos, d_model]
        resid_post = self.hook_resid_post(resid_mid + mlp_out) # [batch, pos, d_model]
        return resid_post

# Full transformer
class Transformer(HookedRootModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        
        self.cfg = cfg
        self.tokenizer = tokenizer
        
        if self.cfg['factored_embed']:
            self.embed = FactoredEmbed(self.cfg)
        else:
            self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint() # [batch, pos, d_model]
        
        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint() # [batch, pos, d_model]

        if cfg['normalization']=='RMS':
            self.norm = RMSNorm(self.cfg, self.cfg['d_model'])
        elif cfg['normalization']=='LN':
            self.norm = LayerNorm(self.cfg, self.cfg['d_model'])
            
        self.blocks = nn.ModuleList([TransformerBlock(self.cfg, block_index) for block_index in range(self.cfg['n_layers'])])

        if self.cfg['factored_embed']:
            self.unembed = FactoredUnembed(self.cfg)
        else:
            self.unembed = Unembed(self.cfg)

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()
            
    def forward(self, tokens, return_type='both', calc_logits=True):
        # Input x is either a batch of tokens ([batch, pos]) or a text string
        if type(tokens)==str:
            # If text, convert to tokens (batch_size=1)
            tokens = self.to_tokens(tokens)
        embed = self.hook_embed(self.embed(tokens)) # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(self.pos_embed(tokens)) # [batch, pos, d_model]
        if cfg['use_pos_resid']:
            residual = embed + pos_embed # [batch, pos, d_model]
        else:
            residual = embed # [batch, pos, d_model]
        for block in self.blocks:
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(residual, pos_embed) # [batch, pos, d_model]
        if not calc_logits:
            # A flag to avoid calculating the logits - this significantly speeds up runtime on small models and reduces memory consumption, and can be used when we only want to get the activations, eg for finding max activating dataset examples.
            return None
        if self.cfg['normalization'] is not None:
            residual = self.norm(residual)
        logits = self.unembed(residual) # [batch, pos, d_vocab]
        if return_type=='both':
            return (logits, loss_fn(logits, tokens))
        elif return_type=='logits':
            return logits
        elif return_type=='loss':
            return loss_fn(logits, tokens)
    
    def to_tokens(self, text):
        return self.tokenizer(self.tokenizer.bos_token+text, return_tensors='pt')['input_ids'].to(self.cfg['device'])


# Transformer Block
class AttnOnlyBlock(nn.Module):
    def __init__(self, cfg, block_index):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)

        self.hook_attn_out = HookPoint() # [batch, pos, d_model]
        # Note that resid_pre of layer k+1 is resid_post of layer k - given for convenience
        self.hook_resid_pre = HookPoint() # [batch, pos, d_model]
        self.hook_resid_post = HookPoint() # [batch, pos, d_model]
    
    def forward(self, x, pos_embed):
        resid_pre = self.hook_resid_pre(x) # [batch, pos, d_model]
        attn_out = self.hook_attn_out(self.attn(x, pos_embed)) # [batch, pos, d_model]
        resid_post = self.hook_resid_post(resid_pre + attn_out) # [batch, pos, d_model]
        return resid_post
        
# Full transformer
class AttnOnlyTransformer(HookedRootModule):
    def __init__(self, cfg, tokenizer):
        raise NotImplementedError("Need to add LN support etc")
        super().__init__()
        
        self.cfg = cfg
        
        self.tokenizer = tokenizer
        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint() # [batch, pos, d_model]
        
        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint() # [batch, pos, d_model]
        
        self.blocks = nn.ModuleList([AttnOnlyBlock(self.cfg, block_index) for block_index in range(self.cfg['n_layers'])])
        self.unembed = Unembed(self.cfg)

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup_hooks()
            
    def forward(self, tokens, return_loss=True):
        # Input x is either a batch of tokens ([batch, pos]) or a text string
        # if type(x)==str:
        #     # If text, convert to tokens (batch_size=1)
        #     x = self.to_tokens(x)
        embed = self.hook_embed(self.embed(tokens)) # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(self.pos_embed(tokens)) # [batch, pos, d_model]
        residual = embed # [batch, pos, d_model]
        for block in self.blocks:
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(residual, pos_embed) # [batch, pos, d_model]
        logits = self.unembed(residual) # [batch, pos, d_vocab]
        if return_loss:
            return loss_fn(logits, tokens)
        else:
            return logits
    
    def to_tokens(self, text):
        return self.tokenizer(text, return_tensors='pt')['input_ids']
# %%
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
    return {'text': np.concatenate([prefix, tokens], axis=1)}# tiny_owt_orig_2 = load_dataset('stas/openwebtext-10k', cache_dir='./cache', split='train', download_config=datasets.DownloadConfig(resume_download=True, num_proc=4))
import time
if False:
    if not cfg['debug']:
        start_time = time.time()
        # randperm = np.random.permutation(28)
        # print('Permutation of PILE URLs', randperm)
        # pile_urls = [f"https://mystic.the-eye.eu/public/AI/pile/train/{i:0>2}.jsonl.zst" for i in randperm]
        # dataset = load_dataset('json', data_files=pile_urls, streaming=True, split='train')

        dataset = load_dataset(cfg['dataset_name'], streaming=True, split='train')
        print('Loaded!', time.time()-start_time)
        start_time = time.time()
        try:
            dataset = dataset.remove_columns('meta')
        except:
            print('Meta not in dataset')
        print('Loaded!', time.time()-start_time)
        start_time = time.time()
        dataset = dataset.map(tokenize, batched=True)
        print('dataset.map', time.time()-start_time)
        start_time = time.time()
        dataset = dataset.with_format(type='torch')
        print('dataset.set_format', time.time()-start_time)
        start_time = time.time()
        dataset = dataset.shuffle(seed=42, buffer_size=100000)
        print('dataset.shuffle', time.time()-start_time)
        start_time = time.time()
        train_data_loader = DataLoader(dataset, batch_size=cfg['batch_size'])
        print('train_data_loader =', time.time()-start_time)
    else:
        streaming_owt = load_dataset('stas/openwebtext-10k', split='train', cache_dir='cache')
        streaming_owt = streaming_owt.map(tokenize, batched=True, num_proc=10)
        streaming_owt = streaming_owt.with_format(type='torch')
        train_data_loader = DataLoader(streaming_owt, batch_size=cfg['batch_size'], shuffle=True)
        start_time = time.time()
        for c, i in tqdm.tqdm(enumerate(train_data_loader)):
            if c == 0:
                print("Loaded Initial stream!")
                print(c, time.time() - start_time)
                start_time = time.time()
            elif c==1:
                print('Time for next batch:', time.time() - start_time)
                break
    data_iter = iter(train_data_loader)
    # tiny_owt_orig_2 = load_dataset('stas/openwebtext-10k', cache_dir='./cache', split='train', download_config=datasets.DownloadConfig(resume_download=True, num_proc=4))
    # print('Loaded!')
    # # tokenizer.add_special_tokens({'pad_token':'<PAD>'})
    # tiny_owt = tiny_owt_orig.map(tokenize, batched=True)
    # print('Tokenized!')
    # tiny_owt_2 = tiny_owt_orig_2.map(tokenize, batched=True)
    print("Running data code!")

# %%
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
pad_token = '<PAD>'
tokenizer.add_special_tokens({'pad_token':pad_token})
print(tokenizer)
# %%
if cfg['attn_only']:
    model = AttnOnlyTransformer(cfg, tokenizer)
else:
    model = Transformer(cfg, tokenizer)
model.to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=cfg['lr'], 
                              betas=cfg['betas'], 
                              weight_decay=cfg['weight_decay'])
# model.load_state_dict(torch.load('/workspace/solu_project/solu_checkpoints/SoLU_1L_v11_final.pth'))
model.load_state_dict(torch.load('/workspace/solu_project/solu_checkpoints/'+cfg['model_checkpoint_name']))
model = (model.to(DTYPE))
print(model)
# %%
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
pad_token = '<PAD>'
tokenizer.add_special_tokens({'pad_token':pad_token})
print(tokenizer)

seq_len = 1024

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
    return {'text': np.concatenate([prefix, tokens], axis=1)}# tiny_owt_orig_2 = load_dataset('stas/openwebtext-10k', cache_dir='./cache', split='train', download_config=datasets.DownloadConfig(resume_download=True, num_proc=4))

# %%

# Define data structure
class MaxActStore():
    length: int
    top_k: int
    max_acts: torch.Tensor
    index: torch.Tensor
    def __init__(self, length, name="", top_k = 10, device='cuda', log=True, log_every=200, dtype=DTYPE):
        self.length = length
        self.top_k = top_k
        self.name = name
        self.log = log
        print("Mapping max acts to dtype", dtype)
        self.max_acts = -torch.inf * torch.ones(length, top_k).to(device).to(dtype)
        # print(self.max_acts)
        self.index = -1 * torch.ones(length, top_k).to(torch.int64).to(device)
        self.counter = 0
        self.total_updates = 0
        self.log_every = log_every
        self.start_time = time.time()
        self.num_steps = 0
    
    def update(self, x: torch.Tensor):
        """Takes in a (length,) tensor of activations

        Args:
            x (torch.Tensor): (length,)
        """
        # print(x.shape, self.max_acts.shape, self.index.shape, self.name, self.counter)
        # print(self.max_acts)
        threshold, lowest_index = self.max_acts.min(1)
        mask = threshold < x
        new_updates = mask.sum().item()
        self.total_updates += new_updates
        if self.log and (self.counter % self.log_every == 0):
            wandb.log({f'{self.name}_max_act': x.max().item(), f'{self.name}_new_updates':new_updates, f'{self.name}_total_updates':self.total_updates}, step=self.counter)
        self.max_acts[mask, lowest_index[mask]] = x[mask].detach()
        self.index[mask, lowest_index[mask]] = self.counter
        self.counter+=1
    
    def batch_update(self, x: torch.Tensor):
        """Takes in a (batch_size, length) tensor of activations

        Args:
            x (torch.Tensor): (batch_size, length)
        """
        # print(self.max_acts)
        acts, indices = x.sort(0, descending=True)
        indices = indices + self.counter
        batch_size = x.size(0)
        new_updates = 0
        for i in range(batch_size):
            threshold, lowest_index = self.max_acts.min(1)
            mask = threshold < acts[i]
            new_updates += mask.sum().item()
            if mask.sum().item() == 0:
                break
            new_updates = mask.sum().item()
            self.total_updates += new_updates
            self.max_acts[mask, lowest_index[mask]] = acts[i][mask].detach()
            self.index[mask, lowest_index[mask]] = indices[i][mask]
        self.total_updates += new_updates
        self.num_steps += i
        if self.log and ((self.counter//batch_size) % self.log_every == 0):
            wandb.log({f'{self.name}_max_act': x.max().item(), f'{self.name}_new_updates':new_updates, f'{self.name}_total_updates':self.total_updates, f'{self.name}_num_steps':self.num_steps, 'elapsed':time.time()-self.start_time}, step=self.counter)
            self.num_steps = 0
        self.counter += batch_size
    
    def set_counter(self, count):
        self.counter = count
    
    def __repr__(self):
        return str((self.counter, self.max_acts, self.index))
    
    def save(self):
        return (self.counter, self.max_acts, self.index)
    def load(self, saved):
        (self.counter, self.max_acts, self.index) = saved

def load_stores_dict(file_name, length, **kwargs):
    saved_dict = torch.load(file_name)
    out = {}
    for k in saved_dict:
        out[k] = MaxActStore(length, **kwargs)
        out[k].load(saved_dict[k])
    return out

def save_stores_dict(stores_dict, file_name, do_wandb=True):
    print(f"Saving stores dict to {file_name}. Available stores are {stores_dict.keys()}")
    obj = {k:v.save() for k, v in stores_dict.items()}
    torch.save(obj, file_name)
    if do_wandb:
        wandb.save(str(file_name))
    
if False:
    wandb.init()
    test_len = 200
    num_tests = 20
    stores_test = {i:MaxActStore(test_len, name='blob'+str(i), log=True, log_every=1) for i in range(num_tests)}
    for i in range(num_tests):
        print(i)
        tor = torch.randn(test_len, 1000).cuda().to(DTYPE)
        # for i in range(1000):
        #     store.update(tor[:, i])
        #     if i % 1000 == 0:
        #         print(i)
        stores_test[i].batch_update(tor.T[:500])
        stores_test[i].batch_update(tor.T[500:])
        val, ind = tor.sort(1, descending=True)
        v2, i2 = (stores_test[i].max_acts.sort(1, descending=True))
        print((v2 == val[:, :10]).all())
        print((stores_test[i].index.gather(-1, i2)==ind[:, :10]).all())
    save_stores_dict(stores_test, 'test.pt', True)
    out_dict = load_stores_dict('test.pt', test_len)
    for i in range(3):
        print(i)
        print((out_dict[i].max_acts == stores_test[i].max_acts).all())
        print((out_dict[i].index == stores_test[i].index).all())
        # print(tor)
    wandb.finish()
# %%
class SaveSchedule():
    def __init__(self, start, scale=1.5):
        self.start = start
        self.scale = scale
        self.next_save = start
        self.counter = 0
    
    def step(self, steps=1):
        self.counter += steps
        if self.counter > self.next_save:
            self.next_save = math.ceil(self.counter * self.scale)
            print(f"Saving at step {self.counter}")
            return True
        return False
if False:
    schedule = SaveSchedule(10)
    t = 0
    for i in range(20):
        k = random.randint(1, 20)
        print(k, t, schedule.step(k), schedule.counter, schedule.next_save)
        t += k
# %%
batch_size = 20
do_logging = True

model.reset_hooks()
def update_store_hook(act, hook, store):
    act_max = act.max(-2).values
    store.batch_update(act_max)

def cache_act(act, hook):
    hook.ctx['act'] = act.detach()

def get_attr_hook(grad, hook, store):
    store.batch_update((grad * hook.ctx['act']).max(-2).values)

stores = {}
for hook in [model.blocks[0].mlp.hook_pre, model.blocks[0].mlp.hook_post, model.blocks[0].mlp.hook_post_ln]:
    s = hook.name.split('.')
    name = f"{s[1]}{s[3][5:]}"
    stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
    hook.add_hook(partial(update_store_hook, store=stores[name]))
    name = f"{s[1]}{s[3][5:]}_attr"
    hook.add_hook(cache_act)
    stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
    hook.add_hook(partial(get_attr_hook, store=stores[name]), dir='bwd')
W_U = model.unembed.W_U
W_out = model.blocks[0].mlp.W_out
# [d_vocab, d_mlp]
if cfg['normalization'] == 'RMS':
    W_logit = W_U @ (model.norm.w[:, None] * W_out)
else:
    W_logit = W_U @ W_out

def direct_logit_attr_hook(act, hook, store):
    # Shape batch, pos, d_mlp
    # act has shape batch, pos, d_mlp
    store.batch_update((act * W_logit[tokens]).max(-2).values)
name = 'logit_attr'
stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
model.blocks[0].mlp.hook_post_ln.add_hook(partial(update_store_hook, store=stores[name]))

# data_loader = DataLoader(dataset, batch_size=20, pin_memory=True)
# batch = next(iter(data_loader))
# tokens = batch['text'].cuda()
# loss = model(tokens, return_type='loss')
# loss.backward()
# del loss
# print(stores)
# if False:
if do_logging:
    wandb.init(config=cfg, project='max_act_solu')
    file_path = ROOT/f"max_act_solu_v2_{str(time.time())}"
    os.mkdir(file_path)
    print("Files will be saved to:", file_path)
data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
data_iter = iter(data_loader)
schedule = SaveSchedule(5*10**4)
if True:
    print("Starting the loop")
    try:
        for c, batch in enumerate(tqdm.tqdm(data_iter)):
            tokens = batch['text'].cuda()
            loss = model(tokens, calc_logits=True, return_type='loss')
            loss.backward()
            if schedule.step(batch_size) and do_logging:
                file_name = file_path/f"{schedule.counter}.pt"
                save_stores_dict(stores, file_name, do_wandb=do_logging)
            del loss
    except KeyboardInterrupt:
        if do_logging:
            file_name = file_path/f"{schedule.counter}.pt"
            save_stores_dict(stores, file_name, do_wandb=do_logging)
    if do_logging:
        wandb.finish()
    print("Finished!")
else:
    print("Skipped the long max examples loop!")
# %%
model.reset_hooks()
# def update_store_hook(act, hook, store):
#     act_max = act.max(-2).values
#     store.batch_update(act_max)

# def cache_act(act, hook):
#     hook.ctx['act'] = act.detach()

# def get_attr_hook(grad, hook, store):
#     store.batch_update((grad * hook.ctx['act']).max(-2).values)

# stores = {}
# for hook in [model.blocks[0].mlp.hook_pre, model.blocks[0].mlp.hook_post, model.blocks[0].mlp.hook_post_ln]:
#     s = hook.name.split('.')
#     name = f"{s[1]}{s[3][5:]}"
#     stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
#     hook.add_hook(partial(update_store_hook, store=stores[name]))
#     name = f"{s[1]}{s[3][5:]}_attr"
#     hook.add_hook(cache_act)
#     stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
#     hook.add_hook(partial(get_attr_hook, store=stores[name]), dir='bwd')
# W_U = model.unembed.W_U
# W_out = model.blocks[0].mlp.W_out
# # [d_vocab, d_mlp]
# if cfg['normalization'] == 'RMS':
#     W_logit = W_U @ (model.norm.w[:, None] * W_out)
# else:
#     W_logit = W_U @ W_out

# def direct_logit_attr_hook(act, hook, store):
#     # Shape batch, pos, d_mlp
#     # act has shape batch, pos, d_mlp
#     store.batch_update((act * W_logit[tokens]).max(-2).values)
# name = 'logit_attr'
# stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
# model.blocks[0].mlp.hook_post_ln.add_hook(partial(update_store_hook, store=stores[name]))

# %%
def text_to_str_tokens(text, tokenizer=tokenizer):
    if text.startswith('<|endoftext|>'):
        return tokenizer.batch_decode(tokenizer.encode(text))
    else:
        return tokenizer.batch_decode(tokenizer.encode("<|endoftext|>"+text))

def vis_activations(str_tokens, activations, name="", incl_bos=True):
    if type(str_tokens)==str:
        str_tokens = text_to_str_tokens(str_tokens)
    if incl_bos:
        pysvelte.TextSingle(tokens=str_tokens, activations=activations[:], neuron_name=name).show()
    else:
        pysvelte.TextSingle(tokens=str_tokens[1:], activations=activations[1:], neuron_name=name).show()
    
def print_neuron_logits(neuron_index, top_k=5):
    l = []
    l.append(f"Top {top_k} logits for Neuron: {neuron_index}")
    logit_vec, logit_indices = W_logit[:, neuron_index].sort()
    for i in range(top_k):
        l.append(f"|{tokenizer.decode([logit_indices[-i-1].item()], clean_up_tokenization_spaces=False)}| {logit_vec[-i-1].item():.6f}")
    l.append('...')
    for i in range(top_k):
        l.append(f"|{tokenizer.decode([logit_indices[top_k-i-1].item()], clean_up_tokenization_spaces=False)}| {logit_vec[top_k-i-1].item():.6f}")
    print("\n".join(l))
print_neuron_logits(0)
to_study = [2, 5, 15, 18]
# sd = torch.load('/workspace/solu_project/max_acts_128W_1_7m_v2.pth')

# for i in sd:
#     val, ind = sd[i][1].sort(1, descending=True)
#     store = MaxActStore
#     sd[i] = (sd[i][0], val, sd[i][2].gather(-1, ind))
print("Loading the stores")
stores_dict = load_stores_dict("/workspace/solu_project/max_act_solu_v2_1662711752.9306793/4328980.pt", cfg['d_mlp'], dtype=torch.float32)

model.reset_hooks()
cache = {}
model.to(torch.float32)
model.cache_all(cache, remove_batch_dim=True)
neuron_index = 0
prefix = 'blocks.0.mlp.hook_'
name = 'post'
names = ['post', 'post_ln', 'pre']
for neuron_index in to_study:
    print_neuron_logits(neuron_index)
    for example_index in range(10):
        text_index = stores_dict['0'+name].index[neuron_index, example_index]
        tokens = dataset[text_index.item()]['text'].unsqueeze(0)
        model(tokens, calc_logits=False)
        text = tokenizer.decode(tokens[0])
        vis_activations(str_tokens=tokenizer.batch_decode(tokens[0]), activations=cache[prefix+name][:, neuron_index], name=f"Act_{name}_N{neuron_index}_#{example_index}")
# %%
