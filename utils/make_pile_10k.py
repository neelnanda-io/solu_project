# %%
from neel.imports import *

pile = datasets.load_dataset('the_pile', split='train', streaming=True)
pile_trunc = pile.take(10**3)
pile_data = list(iter(pile_trunc))
refactor_dict = defaultdict(lambda :[])
for i in range(len(pile_data)):
    for k,v in pile_data[i].items():
        refactor_dict[k].append(v)
trunc_dataset = datasets.Dataset.from_dict(refactor_dict)
print(trunc_dataset)
trunc_dataset.push_to_hub("pile-10k", branch="thousand")

# %%
from huggingface_hub import Repository
from pathlib import Path
repo_root = Path("/workspace/hf_repos")
repo = Repository(local_dir=repo_root/"pile_test", clone_from="datasets/NeelNanda/pile-10k")
print(repo)

# %%



from huggingface_hub import Repository
from pathlib import Path
repo_root = Path("/workspace/hf_repos")
repo = Repository(local_dir=repo_root/"solu_test")
print(repo)
# %%
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path=repo_root/"solu_test/v9_1L",
    path_in_repo="v9_1L",
    repo_id="NeelNanda/SoLU",
    repo_type="model",
)

# %%


# # %%
# from easy_transformer.utils import tokenize_and_concatenate
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

# tok_data = tokenize_and_concatenate(trunc_dataset, tokenizer)
# print(tok_data)
# print(tok_data[0]['tokens'][:100])
# # print(tok_data[0])
# # %%
# print(tokenizer.decode(tok_data[0]['tokens'][:100]))
# # %%
# print(next(iter(pile))['text'][:100])
# print(next(iter(pile_trunc))['text'][:100])
# %%
