# %%
import re
import os
from pathlib import Path
import huggingface_hub
import torch
import json

CACHE_DIR = Path("/workspace/cache")
REPO_ROOT = Path("/workspace/hf_repos/")
CHECKPOINT_DIR = Path("/workspace/solu_project/solu_checkpoints/")

def solu_get_steps(folder_name):
    """Get the list of steps at which checkpoints were taken."""
    file_names = os.listdir(CHECKPOINT_DIR / folder_name)
    big_str = "\n".join(file_names)
    return list(map(int, re.findall("SoLU_\d*L_v\d*_(\d*)\.pth", big_str)))

def solu_get_prev_versions():
    # Extracts version name and layer number from the folder name
    list_pairs = re.findall("v(\d*)_(\d*)L", '\n'.join(os.listdir(CHECKPOINT_DIR)))
    version, layer = zip(*list_pairs)
    return list(map(int, version)), list(map(int, layer))
# %%
def upload_folder_to_hf(folder_name, repo_name, path_in_repo="", create_repo=True, local_root=REPO_ROOT):
    if create_repo:
        try:
            huggingface_hub.create_repo(repo_name)
        except:
            print("Repo already exists")
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
