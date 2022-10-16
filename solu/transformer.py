
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easy_transformer.utils import gelu_new
import einops
from easy_transformer.utils import lm_cross_entropy_loss as loss_fn

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
    log_probs = F.log_softmax(logits[:, :-1].to(torch.float32), dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, batch[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def amp_einsum(einsum_str, mat1, mat2, use_bfloat16=True):
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
        self.W_E = nn.Parameter(torch.empty(
            self.cfg["d_vocab"], self.cfg["d_model"]))
        # nn.init.kaiming_uniform_(self.W_E, a=np.sqrt(5), mode="fan_out")

    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        # return einops.rearrange(self.W_E[tokens, :], 'd_model batch pos -> batch pos d_model')
        return self.W_E[tokens, :]

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty(
            self.cfg["d_model"], self.cfg["d_vocab"]))
        # nn.init.kaiming_uniform_(self.W_U, a=np.sqrt(5), mode="fan_out")

    def forward(self, residual):
        return amp_einsum(
            "bpm,mv->bpv", residual, self.W_U, self.cfg["use_bfloat16_matmul"]
        )  # [batch, pos, d_vocab]

# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(
            self.cfg["n_ctx"], self.cfg["d_model"]))
        # nn.init.kaiming_uniform_(self.W_pos, a=np.sqrt(5), mode="fan_out")

    def forward(self, x):
        # Output shape [pos, d_model] - will be broadcast along batch dim
        return self.W_pos[: x.size(-1), :]  # [pos, d_model]

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

# Attention
class Attention(nn.Module):
    def __init__(self, cfg, attn_type="global"):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(
            torch.empty(self.cfg["n_heads"],
                        self.cfg["d_model"], self.cfg["d_head"])
        )
        self.b_Q = nn.Parameter(torch.zeros(
            self.cfg["n_heads"], self.cfg["d_head"]))
        # nn.init.kaiming_uniform_(self.W_Q, a=np.sqrt(5), mode="fan_out")
        self.W_K = nn.Parameter(
            torch.empty(self.cfg["n_heads"],
                        self.cfg["d_model"], self.cfg["d_head"])
        )
        self.b_K = nn.Parameter(torch.zeros(
            self.cfg["n_heads"], self.cfg["d_head"]))
        # nn.init.kaiming_uniform_(self.W_K, a=np.sqrt(5), mode="fan_out")
        self.W_V = nn.Parameter(
            torch.empty(self.cfg["n_heads"],
                        self.cfg["d_model"], self.cfg["d_head"])
        )
        self.b_V = nn.Parameter(torch.zeros(
            self.cfg["n_heads"], self.cfg["d_head"]))
        # nn.init.kaiming_uniform_(self.W_V, a=np.sqrt(5), mode="fan_out")
        self.W_O = nn.Parameter(
            torch.empty(self.cfg["n_heads"],
                        self.cfg["d_head"], self.cfg["d_model"])
        )
        self.b_O = nn.Parameter(torch.zeros(self.cfg["d_model"]))

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

    def forward(self, resid_pre, attn_input=None):
        if self.cfg["shortformer_pos"]:
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
                    "bpm,imh->bpih", resid_pre, self.W_Q, self.cfg["use_bfloat16_matmul"]
                )
                + self.b_Q
            )  # [batch, pos, head_index, d_head]
            k = self.hook_k(
                amp_einsum(
                    "bpm,imh->bpih", resid_pre, self.W_K, self.cfg["use_bfloat16_matmul"]
                )
                + self.b_K
            )  # [batch, pos, head_index, d_head]

        v = self.hook_v(
            amp_einsum("bpm,imh->bpih", resid_pre, self.W_V,
                       self.cfg["use_bfloat16_matmul"])
            + self.b_V
        )  # [batch, pos, head_index, d_head]
        attn_scores = (
            amp_einsum(
                "bpih,bqih->bipq", q, k, self.cfg["use_bfloat16_matmul"],
            )
            / self.attn_scale
        )  # [batch, head_index, query_pos, key_pos]
        attn_scores = self.hook_attn_scores(
            self.apply_causal_mask(attn_scores)
        )  # [batch, head_index, query_pos, key_pos]
        attn_matrix = self.hook_attn(
            F.softmax(attn_scores.to(torch.float32), dim=-1)
        )  # [batch, head_index, query_pos, key_pos]
        z = self.hook_z(
            amp_einsum(
                "bpih,biqp->bqih", v, attn_matrix, self.cfg["use_bfloat16_matmul"]
            )
        )  # [batch, pos, head_index, d_head]

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
        self.W_in = nn.Parameter(torch.empty(
            self.cfg["d_model"], self.cfg["d_mlp"]))
        # nn.init.kaiming_uniform_(self.W_in, a=np.sqrt(5), mode="fan_out")
        self.b_in = nn.Parameter(torch.zeros(self.cfg["d_mlp"]))
        self.W_out = nn.Parameter(torch.empty(
            self.cfg["d_mlp"], self.cfg["d_model"]))
        # nn.init.kaiming_uniform_(self.W_out, a=np.sqrt(5), mode="fan_out")
        self.b_out = nn.Parameter(torch.zeros(self.cfg["d_model"]))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg["act_fn"].lower() == "gelu":
            self.act_fn = gelu_new
        elif self.cfg["act_fn"].lower() == "solu_ln":
            self.act_fn = lambda x: F.softmax(x.to(torch.float32), dim=-1) * x
            self.hook_post_ln = HookPoint()  # [batch, pos, d_mlp]
            self.ln = LayerNorm(self.cfg, self.cfg["d_mlp"])
        else:
            raise ValueError(
                f"Invalid activation function name: {self.cfg['act_fn']}")

    def forward(self, x):
        x = self.hook_pre(
            amp_einsum("bpd,dm->bpm", x, self.W_in,
                       self.cfg["use_bfloat16_matmul"])
            + self.b_in
        )  # [batch, pos, d_mlp]
        x = self.hook_post(self.act_fn(x/self.cfg["neuron_temp"]))  # [batch, pos, d_mlp]
        if self.cfg["act_fn"].lower() == "solu_ln":
            x = self.hook_post_ln(self.ln(x))
        x = (
            amp_einsum("bpm,md->bpd", x, self.W_out,
                       self.cfg["use_bfloat16_matmul"])
            + self.b_out
        )  # [batch, pos, d_model]
        return x * self.cfg["neuron_scale"]


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg, block_index):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(self.cfg, self.cfg["d_model"])
        self.attn = Attention(self.cfg)
        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]

        if cfg['shortformer_pos']:
            self.hook_attn_input = HookPoint()  # [batch, pos, d_model]
        # Note that resid_pre of layer k+1 is resid_post of layer k - given for convenience
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]
        if not cfg["attn_only"]:
            self.ln2 = LayerNorm(self.cfg, self.cfg["d_model"])
            self.mlp = MLP(self.cfg)
            self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
            self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
    def forward(self, x, pos_embed):
        resid_pre = self.hook_resid_pre(x)  # [batch, pos, d_model]
        if self.cfg['shortformer_pos']:
            attn_input = self.hook_attn_input(resid_pre+pos_embed)
            attn_out = self.hook_attn_out(
                self.attn(self.ln1(resid_pre), self.ln1(attn_input))
            )  # [batch, pos, d_model]
        else:
            attn_out = self.hook_attn_out(
                self.attn(self.ln1(resid_pre))
            )  # [batch, pos, d_model]
        if self.cfg["attn_only"]:
            resid_post = self.hook_resid_post(
                resid_pre + attn_out)  # [batch, pos, d_model]
        else:
            resid_mid = self.hook_resid_mid(
                resid_pre + attn_out)  # [batch, pos, d_model]
            
            mlp_out = self.hook_mlp_out(
                self.mlp(self.ln2(resid_mid))
            )  # [batch, pos, d_model]
            resid_post = self.hook_resid_post(
                resid_mid + mlp_out)  # [batch, pos, d_model]
        return resid_post


# Full transformer
class Transformer(HookedRootModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint()  # [batch, pos, d_model]

        self.ln_final = LayerNorm(self.cfg, self.cfg["d_model"])

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
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(
            self.pos_embed(tokens))  # [batch, pos, d_model]
        if not self.cfg["shortformer_pos"]:
            residual = embed + pos_embed  # [batch, pos, d_model]
        else:
            residual = embed  # [batch, pos, d_model]
        for block in self.blocks:
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(residual, pos_embed)  # [batch, pos, d_model]
        
        residual = self.ln_final(residual)
        logits = self.unembed(residual.to(torch.float32))  # [batch, pos, d_vocab]

        if return_loss:
            return loss_fn(logits, tokens)
        else:
            return logits
