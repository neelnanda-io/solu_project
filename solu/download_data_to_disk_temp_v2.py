# %%
from neel.imports import *
n = 30

c4_urls = [f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:0>5}-of-01024.json.gz" for i in range(n)]

dataset = load_dataset('json', data_files=c4_urls, split='train')

print(dataset)
# %%

from easy_transformer.utils import tokenize_and_concatenate
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# full_cc_data = datasets.load_from_disk(f'/workspace/data/c4_train_160_text.hf')
# print("Loaded data")
# print(full_cc_data)
# # %%
# full_cc_data = 
# dataset = 
tokens_cc = tokenize_and_concatenate(dataset, tokenizer, streaming=False)

tokens_cc.save_to_disk(f'/workspace/data/c4_train_{n}_tokens_gpt2.hf')

print(tokens_cc)

# %%
# mode = "train"
# code_data = load_dataset(f"codeparrot/codeparrot-{mode}-v2-near-dedup", split="train")

# code_data = code_data.train_test_split(0.05)
# code_data.save_to_disk(f"/workspace/data/codeparrot_{mode}_split.hf")
# code_data_train = code_data['train']

# tokens_code = tokenize_and_concatenate(code_data_train, tokenizer, streaming=False, column_name="content")
# tokens_code.save_to_disk(f'/workspace/data/codeparrot_{mode}_tokens.hf')
# # %%
