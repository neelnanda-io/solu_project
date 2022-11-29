# %%
import einops
import re
import os
from pathlib import Path
import huggingface_hub
import torch
import json
import numpy as np
import plotly.express as px
import logging
import shutil
import argparse
import datasets
from typing import Tuple, Union
from IPython import get_ipython
from easy_transformer.utils import get_corner

CACHE_DIR = Path.home() / ("cache")
REPO_ROOT = Path.home() / ("hf_repos/")
OLD_CHECKPOINT_DIR = Path.home() / ("solu_project/solu_checkpoints/")
CHECKPOINT_DIR = Path.home() / ("solu_project/saved_models/")


def solu_get_steps(folder_name):
    """Get the list of steps at which checkpoints were taken."""
    file_names = os.listdir(OLD_CHECKPOINT_DIR / folder_name)
    big_str = "\n".join(file_names)
    return list(map(int, re.findall("SoLU_\d*L_v\d*_(\d*)\.pth", big_str)))


def solu_get_prev_versions_old():
    # Extracts version name and layer number from the folder name
    list_pairs = re.findall("v(\d*)_(\d*)L", "\n".join(os.listdir(OLD_CHECKPOINT_DIR)))
    version, layer = zip(*list_pairs)
    return list(map(int, version)), list(map(int, layer))


def solu_get_prev_versions():
    # Extracts version name and layer number from the folder name
    versions = re.findall("v(\d*)_.*", "\n".join(os.listdir(CHECKPOINT_DIR)))
    return list(map(int, versions))


# %%
# def upload_folder_to_hf(
#     folder_name, repo_name, path_in_repo="", create_repo=True, local_root=REPO_ROOT
# ):
#     if create_repo:
#         try:
#             huggingface_hub.create_repo(repo_name)
#         except Exception as e:
#             print("Repo already exists")
#             print("Error:", e)
#     api = huggingface_hub.HfApi()
#     link = api.upload_folder(
#         folder_path=local_root / folder_name,
#         path_in_repo=path_in_repo,
#         repo_id=f"NeelNanda/{repo_name}",
#     )
#     print(f"Successfully uploaded to: {link}")


# def upload_file_to_hf(
#     file_name, repo_name, path_in_repo="", create_repo=True, local_root=REPO_ROOT
# ):
#     # Note - does not currently seem to work :( Not sure why.
#     if create_repo:
#         try:
#             huggingface_hub.create_repo(repo_name)
#         except:
#             print("Repo already exists")
#     api = huggingface_hub.HfApi()
#     link = api.upload_file(
#         path_or_fileobj=str(local_root / file_name),
#         path_in_repo=path_in_repo,
#         repo_id=f"NeelNanda/{repo_name}",
#     )
#     print(f"Successfully uploaded to: {link}")


def download_file_from_hf(repo_name, file_name, subfolder=".", cache_dir=CACHE_DIR):
    file_path = huggingface_hub.hf_hub_download(
        repo_id=f"NeelNanda/{repo_name}",
        filename=file_name,
        subfolder=subfolder,
        cache_dir=cache_dir,
    )
    print(f"Saved at file_path: {file_path}")
    if file_path.endswith(".pth"):
        return torch.load(file_path)
    elif file_path.endswith(".json"):
        return json.load(open(file_path, "r"))
    else:
        print("File type not supported:", file_path.split(".")[-1])
        return file_path


# %%


class SaveSchedule:
    def __init__(self, max_tokens, tokens_per_step, schedule=None, show_plot=False):
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
        if show_plot:
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


# %%
def push_to_hub(local_dir):
    if isinstance(local_dir, huggingface_hub.Repository):
        local_dir = local_dir.local_dir
    os.system(f"git -C {local_dir} add .")
    os.system(f"git -C {local_dir} commit -m 'Auto Commit'")
    os.system(f"git -C {local_dir} push")


# move_folder_to_hub("v235_4L512W_solu_wikipedia", "NeelNanda/SoLU_4L512W_Wiki_Finetune", just_final=False)
def move_folder_to_hub(model_name, repo_name=None, just_final=True, debug=False):
    if repo_name is None:
        repo_name = model_name
    model_folder = CHECKPOINT_DIR / model_name
    repo_folder = CHECKPOINT_DIR / (model_name + "_repo")
    repo_url = huggingface_hub.create_repo(repo_name, exist_ok=True)
    repo = huggingface_hub.Repository(repo_folder, repo_url)

    for file in model_folder.iterdir():
        if not just_final or "final" in file.name or "config" in file.name:
            if debug:
                print(file.name)
            file.rename(repo_folder / file.name)
    push_to_hub(repo.local_dir)
def upload_folder_to_hf(folder_path, repo_name=None, debug=False):
    folder_path = Path(folder_path)
    if repo_name is None:
        repo_name = folder_path.name
    repo_folder = folder_path.parent / (folder_path.name + "_repo")
    repo_url = huggingface_hub.create_repo(repo_name, exist_ok=True)
    repo = huggingface_hub.Repository(repo_folder, repo_url)

    for file in folder_path.iterdir():
        if debug:
            print(file.name)
        file.rename(repo_folder / file.name)
    push_to_hub(repo.local_dir)


# %%
# Model diagnostics


def print_param_norms(model):
    for name, param in model.named_parameters():
        if "IGNORE" not in name and "mask" not in name:
            print(name, param.norm().item())


def print_act_norms(model, tokens):
    cache = {}
    model.cache_all(cache)
    loss = model(tokens)
    print("Loss:", loss.item())
    for key, value in cache.items():
        print(key, value.norm())


# %%
def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg


# %%
def get_dataset(dataset_name) -> Union[datasets.Dataset, datasets.DatasetDict]:
    if dataset_name == "c4-code":
        # Pre shuffled!
        tokens_c4_code = datasets.load_from_disk(
            "/workspace/data/c4_code_valid_tokens.hf"
        )
        tokens_c4_code = tokens_c4_code.with_format("torch")
        return tokens_c4_code
    elif dataset_name == "c4":
        tokens_c4 = datasets.load_from_disk(f"/workspace/data/c4_valid_tokens.hf")
        tokens_c4 = tokens_c4.with_format("torch")
        return tokens_c4
    elif dataset_name == "code":
        tokens_code = datasets.load_from_disk(f"/workspace/data/code_valid_tokens.hf")
        tokens_code = tokens_code.with_format("torch")
        return tokens_code
    elif dataset_name == "pile":
        # Pre shuffled
        tokens_pile = datasets.load_from_disk("/workspace/data/pile_valid_tokens.hf")
        tokens_pile = tokens_pile.with_format("torch")
        return tokens_pile
    elif dataset_name == "openwebtext":
        # Pre shuffled
        tokens_owt = datasets.load_from_disk("/workspace/data/openwebtext_tokens.hf")
        tokens_owt = tokens_owt.with_format("torch")
        return tokens_owt


def get_next_version(DIR):
    # Extracts version name and layer number from the folder name
    if not DIR.exists():
        return 0
    versions = re.findall("v(\d*)", "\n".join(os.listdir(DIR)))
    versions = list(map(int, versions))
    print(versions)
    if versions:
        return max(versions) + 1
    else:
        return 0


""" 
Test:
ms = MaxStore(3, 100)

big_arr = torch.randn(10**5, 100).cuda()
for i in range(0, 10**5, 5):
    ms.batch_update(big_arr[i:i+5])
ms.switch_to_inference()
sortv, sorti = big_arr.sort(dim=0, descending=True)
print(ms.index[5])
print(sorti[:3, 5])
print(ms.max[5])
print(sortv[:3, 5])
"""


class MaxStore:
    """Used to calculate max activating dataset examples - takes in batches of activations repeatedly, and tracks the top_k examples activations + indexes"""

    def __init__(self, top_k, length, device="cuda"):
        self.top_k = top_k
        self.length = length
        self.device = device

        self.max = -torch.inf * torch.ones(
            (top_k, length), dtype=torch.float32, device=device
        )
        self.index = -torch.ones((top_k, length), dtype=torch.long, device=device)

        self.counter = 0
        self.total_updates = 0
        self.batches_seen = 0

    def update(self, new_act, new_index):
        min_max_act, min_indices = self.max.min(dim=0)
        mask = min_max_act < new_act
        num_updates = mask.sum().item()
        self.max[min_indices[mask], mask] = new_act[mask]
        self.index[min_indices[mask], mask] = new_index[mask]
        self.total_updates += num_updates
        return num_updates

    def batch_update(self, activations, text_indices=None):
        """
        activations: Shape [batch, length]
        text_indices: Shape [batch,]

        activations is the largest MLP activation, text_indices is the index of the text strings.

        Sorts the activations into descending order, then updates with each column until we stop needing to update
        """
        batch_size = activations.size(0)
        new_acts, sorted_indices = activations.sort(0, descending=True)
        if text_indices is None:
            text_indices = torch.arange(
                self.counter,
                self.counter + batch_size,
                device=self.device,
                dtype=torch.int64,
            )
        new_indices = text_indices[sorted_indices]
        for i in range(batch_size):
            num_updates = self.update(new_acts[i], new_indices[i])
            if num_updates == 0:
                break
        self.counter += batch_size
        self.batches_seen += 1

    def save(self, dir, folder_name=None):
        if folder_name is not None:
            path = dir / folder_name
        else:
            path = dir
        path.mkdir(exist_ok=True)
        torch.save(self.max, path / "max.pth")
        torch.save(self.index, path / "index.pth")
        with open(path / "config.json", "w") as f:
            filt_dict = {
                k: v for k, v in self.__dict__.items() if k not in ["max", "index"]
            }
            json.dump(filt_dict, f)
        print("Saved Max Store to:", path)

    def switch_to_inference(self):
        """Switch from updating mode to inference - move to the CPU and sort by max act."""
        self.max = self.max.cpu()
        self.index = self.index.cpu()
        self.max, indices = self.max.sort(dim=0, descending=True)
        self.index = self.index.gather(0, indices)

    @classmethod
    def load(cls, dir, folder_name=None, continue_updating=False, transpose=False):
        dir = Path(dir)
        if folder_name is not None:
            path = dir / folder_name
        else:
            path = dir

        max = torch.load(path / "max.pth")
        index = torch.load(path / "index.pth")
        if transpose:
            max = max.T
            index = index.T
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        mas = cls(config["top_k"], config["length"])
        for k, v in config.items():
            mas.__dict__[k] = v
        mas.max = max
        mas.index = index
        if not continue_updating:
            mas.switch_to_inference()
        return mas

    def __repr__(self):
        return f"MaxActStore(top_k={self.top_k}, length={self.length}, counter={self.counter}, total_updates={self.total_updates}, device={self.device})\n Max Acts: {get_corner(self.max)}\n Indices: {get_corner(self.index)}"
