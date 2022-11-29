from neel.imports import *
from solu.utils import *

folder_names = ["v23_12L", "v22_10L", "v21_8L", "v28_2L", "v10_2L", "v9_1L"]
for folder in reversed(folder_names):
    version, layer = re.match("v(\d*)_(\d*)L", folder).groups(0)
    repo_name = f"SoLU_{layer}L_v{version}_old"
    move_folder_to_hub(folder, repo_name, just_final=False)