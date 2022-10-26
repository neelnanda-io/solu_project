# %%
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

CACHE_DIR = Path.home()/("cache")
REPO_ROOT = Path.home()/("hf_repos/")
OLD_CHECKPOINT_DIR = Path.home()/("solu_project/solu_checkpoints/")
CHECKPOINT_DIR = Path.home()/("solu_project/saved_models/")

def solu_get_steps(folder_name):
    """Get the list of steps at which checkpoints were taken."""
    file_names = os.listdir(OLD_CHECKPOINT_DIR / folder_name)
    big_str = "\n".join(file_names)
    return list(map(int, re.findall("SoLU_\d*L_v\d*_(\d*)\.pth", big_str)))

def solu_get_prev_versions_old():
    # Extracts version name and layer number from the folder name
    list_pairs = re.findall("v(\d*)_(\d*)L", '\n'.join(os.listdir(OLD_CHECKPOINT_DIR)))
    version, layer = zip(*list_pairs)
    return list(map(int, version)), list(map(int, layer))

def solu_get_prev_versions():
    # Extracts version name and layer number from the folder name
    versions = re.findall("v(\d*)_.*", '\n'.join(os.listdir(CHECKPOINT_DIR)))
    return list(map(int, versions))
# %%
def upload_folder_to_hf(folder_name, repo_name, path_in_repo="", create_repo=True, local_root=REPO_ROOT):
    if create_repo:
        try:
            huggingface_hub.create_repo(repo_name)
        except Exception as e:
            print("Repo already exists")
            print("Error:", e)
    api = huggingface_hub.HfApi()
    link = api.upload_folder(
        folder_path=local_root/folder_name,
        path_in_repo=path_in_repo,
        repo_id=f"NeelNanda/{repo_name}",
    )
    print(f"Successfully uploaded to: {link}")

def upload_file_to_hf(file_name, repo_name, path_in_repo="", create_repo=True, local_root=REPO_ROOT):
    # Note - does not currently seem to work :( Not sure why.
    if create_repo:
        try:
            huggingface_hub.create_repo(repo_name)
        except:
            print("Repo already exists")
    api = huggingface_hub.HfApi()
    link = api.upload_file(
        path_or_fileobj=str(local_root/file_name),
        path_in_repo=path_in_repo,
        repo_id=f"NeelNanda/{repo_name}",
    )
    print(f"Successfully uploaded to: {link}")

def download_file_from_hf(repo_name, file_name, subfolder=".", cache_dir=CACHE_DIR):
    file_path = huggingface_hub.hf_hub_download(repo_id=f"NeelNanda/{repo_name}",
                                                filename=file_name, 
                                                subfolder=subfolder, 
                                                cache_dir=cache_dir)
    print(f"Saved at file_path: {file_path}")
    if file_path.endswith(".pth"):
        return torch.load(file_path)
    elif file_path.endswith(".json"):
        return json.load(open(file_path, "r"))
    else:
        print("File type not supported:", file_path.split('.')[-1])
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

def move_folder_to_hub(model_name, repo_name=None, just_final=True, debug=False):
    if repo_name is None:
        repo_name = model_name
    model_folder = CHECKPOINT_DIR / model_name
    repo_folder = CHECKPOINT_DIR / (model_name+"_repo")
    repo_url = huggingface_hub.create_repo(repo_name, exist_ok=True)
    repo = huggingface_hub.Repository(repo_folder, repo_url)
    
    for file in model_folder.iterdir():
        if not just_final or "final" in file.name or "config" in file.name:
            if debug: print(file.name)
            file.rename(repo_folder/file.name)
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
