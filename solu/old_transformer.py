from easy_transformer.hook_points import HookPoint, HookedRootModule
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random

from easy_transformer.utils import gelu_new
from easy_transformer.utils import lm_cross_entropy_loss as loss_fn
import einops

# Define network architecture


def amp_einsum(einsum_str, mat1, mat2, use_bfloat16=True):
    # return torch.einsum(einsum_str, mat1, mat2)
    # return torch.einsum(einsum_str, mat1.to(torch.bfloat16), mat2.to(torch.bfloat16)).to(torch.float32)
    if use_bfloat16:
        return torch.einsum(
            einsum_str, mat1.to(torch.bfloat16), mat2.to(torch.bfloat16)
        )
    else:
        return torch.einsum(einsum_str, mat1, mat2)


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
            "bpm,mv->bpv",
            residual,
            self.W_U,
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
        # [batch, head_index, query_pos, key_pos]
        self.hook_attn_scores = HookPoint()
        self.hook_attn = HookPoint()  # [batch, head_index, query_pos, key_pos]
        # [batch, head_index, head_index, d_model]
        self.hook_result = HookPoint()
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
                )
                + self.b_Q
            )  # [batch, pos, head_index, d_head]
            k = self.hook_k(
                amp_einsum(
                    "bpm,imh->bpih",
                    attn_input,
                    self.W_K,
                )
                + self.b_K
            )  # [batch, pos, head_index, d_head]
        else:
            q = self.hook_q(
                amp_einsum("bpm,imh->bpih", x, self.W_Q) + self.b_Q
            )  # [batch, pos, head_index, d_head]
            k = self.hook_k(
                amp_einsum("bpm,imh->bpih", x, self.W_K) + self.b_K
            )  # [batch, pos, head_index, d_head]

        v = self.hook_v(
            amp_einsum("bpm,imh->bpih", x, self.W_V) + self.b_V
        )  # [batch, pos, head_index, d_head]
        attn_scores = (
            amp_einsum(
                "bpih,bqih->bipq",
                q.to(torch.float32),
                k.to(torch.float32),
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
            amp_einsum("bpih,biqp->bqih", v, attn_matrix)
        )  # [batch, pos, head_index, d_head]

        if self.cfg["use_attn_result"]:
            result = self.hook_result(
                amp_einsum("bqih,ihm->bqim", z, self.W_O)
            )  # [batch, pos, head_index, d_model]
            out = result.sum(-2) + self.b_O  # [batch, pos, d_model]
        else:
            out = (
                amp_einsum("bqih,ihm->bqm", z, self.W_O) + self.b_O
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
            amp_einsum("bpd,dm->bpm", x, self.W_in) + self.b_in
        )  # [batch, pos, d_mlp]
        x = self.hook_post(self.act_fn(x))  # [batch, pos, d_mlp]
        if self.cfg["act_fn"].lower() == "solu":
            x = self.hook_post_ln(self.ln(x))
        x = (
            amp_einsum("bpm,md->bpd", x, self.W_out) + self.b_out
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
