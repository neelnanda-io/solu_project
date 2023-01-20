# %%
from neel.imports import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
args = parser.parse_args()
args = vars(args)
dataset_name = args["data"]
tokenizer_name = "gpt2"
print(dataset_name, tokenizer_name)
# parser.add_argument('--start', type=int, default=0)
# parser.add_argument('--end', type=int, default=160)
# n = 640
# n = 160

# c4_urls = [f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:0>5}-of-01024.json.gz" for i in range(n)]

dataset = load_dataset(dataset_name, split="train")
# dataset = load_dataset("wikipedia", "20220301.en", split="train")
# print(dataset)
dataset.save_to_disk(f"/workspace/data/{dataset_name}_text.hf")
# %%
# dataset = datasets.load_from_disk("/workspace/data/openwebtext_text.hf")
print(dataset)

from transformer_lens.utils import tokenize_and_concatenate

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokens = tokenize_and_concatenate(dataset, tokenizer, streaming=False, num_proc=20)
tokens.save_to_disk(f"/workspace/data/{dataset_name}_tokens.hf")
# tokens.save_to_disk(f'/workspace/data/wiki_train_tokens.hf')
# tokens.save_to_disk(f'/workspace/data/c4_{n}_train_tokens.hf')
# tokens.save_to_disk(f'/workspace/data/c4_valid_tokens.hf')


# full_cc_data = datasets.load_from_disk(f'/workspace/data/c4_train_160_text.hf')
# print("Loaded data")
# print(full_cc_data)
# # %%
# full_cc_data =
# dataset =
# tokens_owt = tokenize_and_concatenate(dataset, tokenizer, streaming=False)

# tokens_owt.save_to_disk(f'/workspace/data/owt_tokens_gpt2.hf')

# print(tokens_owt)

# %%
# mode = "train"
# code_data = load_dataset(f"codeparrot/codeparrot-{mode}-v2-near-dedup", split="train")

# code_data = code_data.train_test_split(0.05)
# code_data.save_to_disk(f"/workspace/data/codeparrot_{mode}_split.hf")
# code_data_train = code_data['train']

# tokens_code = tokenize_and_concatenate(code_data_train, tokenizer, streaming=False, column_name="content")
# tokens_code.save_to_disk(f'/workspace/data/codeparrot_{mode}_tokens.hf')
# # %%
