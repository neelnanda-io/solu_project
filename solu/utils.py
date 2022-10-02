# %%
import re
import os
from pathlib import Path

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
