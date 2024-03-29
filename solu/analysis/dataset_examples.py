# %%
# Imports
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
import sys
from pprint import pprint

import pysvelte
from transformer_lens import *

if "ipykernel_launcher" in os.path.basename(sys.argv[0]):
    %matplotlib inline
    %load_ext autoreload
    %autoreload 2
    print("Activated reload")
# %%
# Set config
cfg = {
    'd_model':1024,
    # 'n_heads':8,
    'd_head':64,
    'n_layers':1,
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
    'normalization':'RMS', # 'LN' 'RMS' or None
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
# Plotting functions
if True:
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

# %%
# Defining the model
if True:
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
            scale = self.hook_scale((einops.reduce(x.pow(2), 
                                                'batch pos embed -> batch pos 1', 
                                                'mean') + 
                                    self.eps).sqrt()) # [batch, pos, 1]
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
            scale = self.hook_scale((einops.reduce(x.pow(2), 
                                                'batch pos embed -> batch pos 1', 
                                                'mean') + 
                                    self.eps).sqrt()) # [batch, pos, 1]
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
            scale = self.hook_scale((einops.reduce(x.pow(2), 
                                                'batch pos embed -> batch pos 1', 
                                                'mean') + 
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
                
        def forward(self, tokens, return_type='both'):
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
def loss_fn(logits, batch):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, batch[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()

def per_token_loss_fn(logits, batch):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, batch[:, 1:, None])[..., 0]
    return -pred_log_probs
# %%
# Load tokenizer
if True:
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    pad_token = '<PAD>'
    tokenizer.add_special_tokens({'pad_token':pad_token})
    print(tokenizer)
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

# %%
# Tokenize data

# %%
# Load model
if cfg['attn_only']:
    model = AttnOnlyTransformer(cfg, tokenizer)
else:
    model = Transformer(cfg, tokenizer)
model.load_state_dict(torch.load(Path.home()/'solu_project/solu_checkpoints/SoLU_1L_v9_final.pth'))
model.to('cuda')

# %%
# Create hooks - these will take in the neuron activations and create a running queue of the best activations so far, and the resulting activation index. Iterate through these one at a time, I think?