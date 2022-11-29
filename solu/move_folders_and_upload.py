# %%
from pathlib import Path
import os
from solu.utils import *
from huggingface_hub import create_repo, Repository
from tqdm.auto import tqdm

old_checkpoint_dir = Path.home() / ("solu_project/solu_checkpoints/")
new_checkpoint_dir = Path.home() / ("solu_project/saved_models/")


# OK let's structure the code:
# Iterate over folders

# for folder in old_checkpoint_dir.glob("v*"):
#     print(folder)

#     version_name = folder.stem
#     version, layer = version_name.split("_")
#     model_name = f"SoLU_{layer}_{version}"
#     initial_file_path = folder/f"{model_name}_000000.pth"
#     if initial_file_path.exists():
#         print("Changing", initial_file_path, "to", folder/f"{model_name}_init.pth")
#         initial_file_path.rename(folder/f"{model_name}_init.pth")

#     sub_checkpoints_dir = folder/"checkpoints"
#     sub_checkpoints_dir.mkdir()
#     for file_path in folder.glob(f"{model_name}_0*"):
#         file_path.rename(sub_checkpoints_dir/file_path.name)
#     print("Moved checkpoints")

#     # Move to saved_models
#     folder.rename(new_checkpoint_dir/version_name)
#     print("Moved folder")
#     # Upload the entire directory, with model name
#     # upload_folder_to_hf(f"{version_name}", f"{model_name}", create_repo=True, local_root=new_checkpoint_dir)
#     # print("Uploaded to HF")

# %%
folders = ["v11_4L", "v13_6L", "v21_8L", "v22_10L", "v23_12L"]

for folder in tqdm(folders):
    print("New path")
    path = new_checkpoint_dir / folder
    print(path)
    version, layers = map(int, re.match("v(\d*)_(\d*)L", path.name).groups(0))
    print(version, layers)
    model_name = f"SoLU_{layers}L_v{version}"
    print(model_name)

    repo_url = create_repo(model_name + "_old", exist_ok=True)
    repo_path = path.parent / (path.name + "_repo")
    repo = Repository(repo_path, clone_from=repo_url)
    content_paths = path.glob("*")
    for content_path in content_paths:
        content_path.rename(repo_path / content_path.name)

    print("Starting to push!")
    repo.push_to_hub(blocking=False)

# %%
