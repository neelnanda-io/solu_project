# %%
import os
import time
import datasets
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import einops 
import argparse
import datasets
from datasets import disable_caching
disable_caching()
datasets.config.IN_MEMORY_MAX_SIZE = 3*10**11
import time
start_time = time.time()

c4_urls = [f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:0>5}-of-01024.json.gz" for i in range(200)]


dataset = load_dataset('json', data_files=c4_urls, split='train')
print(dataset)
print(dataset[0])
print('Loaded!', time.time()-start_time)
# for key in dataset.features:
#     if key != "text":
#         print("Deleting feature", key)
#         dataset = dataset.remove_columns(key)
# print("New dataset")
# print(dataset)
# # %%
# tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
# pad_token = '<PAD>'
# tokenizer.add_special_tokens({'pad_token':pad_token})
# print(tokenizer)

# seq_len = 1024
# # %%
# def tokenize(examples):
#     texts = examples['text']
#     full_text = tokenizer.eos_token.join(texts)
#     div = 20
#     length = len(full_text)//div
#     text_list = [full_text[i*length:(i+1)*length] for i in range(div)]
#     tokens = tokenizer(text_list, return_tensors='np', padding=True)['input_ids'].flatten()
#     tokens = tokens[tokens!=tokenizer.pad_token_id]
    
#     n = len(tokens)
#     curr_batch_size = n//(seq_len-1)
#     tokens = tokens[:(seq_len-1)*curr_batch_size]
#     tokens = einops.rearrange(tokens, '(batch_size seq) -> batch_size seq', batch_size=curr_batch_size, seq=seq_len-1)
#     prefix = np.ones((curr_batch_size, 1), dtype=np.int64)*tokenizer.bos_token_id
#     return {'text': np.concatenate([prefix, tokens], axis=1)}
# print("Starting to map:")
# start_time = time.time()
# dataset = dataset.map(tokenize, batched=True, num_proc=20, keep_in_memory=True)
# print('dataset.map', time.time()-start_time)
# dataset = dataset.with_format(type='torch')
# print("Set torch!")
# if not args.pile:
#     file_path = Path.home()/f"data/c4_{args.Index:0>5}.hf"
# else:
#     file_path = Path.home()/f"data/pile_{args.Index:0>2}.hf"
# print("Saving to file", file_path)
# dataset.save_to_disk(file_path)
# print("Number of removed cache files:", dataset.cleanup_cache_files())
# # %%
# print("Time taken:", time.time()-start_time)
# os.system("rm ~/cache/* -r")
