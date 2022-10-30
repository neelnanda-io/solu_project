# %%
from neel.imports import *
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--start', type=int, default=0)
# parser.add_argument('--end', type=int, default=160)
n = 640
# n = 160

c4_urls = [f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:0>5}-of-01024.json.gz" for i in range(n)]

dataset = load_dataset('json', data_files=c4_urls, split='train')
# dataset = load_dataset("wikipedia", "20220301.en", split="train")
print(dataset)
# %%

from easy_transformer.utils import tokenize_and_concatenate
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")

tokens = tokenize_and_concatenate(dataset, tokenizer, streaming=False)
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
