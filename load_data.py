# %%
import time
import datasets
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import einops
import argparse

# Python program to demonstrate
# command line arguments
 
 
import argparse
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("Index", help = "Index of Pile File", type=int, choices=range(30))
 
# Read arguments from command line
args = parser.parse_args()
 
if args.Index:
    print(f"Index: {args.Index}")
# print(1/0)

start_time = time.time()
# if cfg['shuffled_data']:
# randperm = np.random.permutation(28)
# print('Permutation of PILE URLs', randperm)
# pile_urls = [f"https://mystic.the-eye.eu/public/AI/pile/train/{i:0>2}.jsonl.zst" for i in randperm]
pile_url = "https://mystic.the-eye.eu/public/AI/pile/train/{args.Index:0>2}.jsonl.zst"
print("Reading from the Pile from", pile_url)
dataset = load_dataset('json', data_files=pile_url, split='train', keep_in_memory=True)
print(dataset[0])
    # else:
    #     dataset = load_dataset(cfg['dataset_name'], streaming=True, split='train')
print('Loaded!', time.time()-start_time)
try:
    dataset = dataset.remove_columns('meta')
except:
    print('Meta not in dataset')
# %%
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
pad_token = '<PAD>'
tokenizer.add_special_tokens({'pad_token':pad_token})
print(tokenizer)

seq_len = 1024
# %%
def tokenize(examples):
    texts = examples['text']
    full_text = tokenizer.eos_token.join(texts)
    div = 20
    length = len(full_text)//div
    text_list = [full_text[i*length:(i+1)*length] for i in range(div)]
    tokens = tokenizer(text_list, return_tensors='np', padding=True)['input_ids'].flatten()
    tokens = tokens[tokens!=tokenizer.pad_token_id]
    
    n = len(tokens)
    curr_batch_size = n//(seq_len-1)
    tokens = tokens[:(seq_len-1)*curr_batch_size]
    tokens = einops.rearrange(tokens, '(batch_size seq) -> batch_size seq', batch_size=curr_batch_size, seq=seq_len-1)
    prefix = np.ones((curr_batch_size, 1), dtype=np.int64)*tokenizer.bos_token_id
    return {'text': np.concatenate([prefix, tokens], axis=1)}
print("Starting to map:")
start_time = time.time()
dataset = dataset.map(tokenize, batched=True, num_proc=20, keep_in_memory=True)
print('dataset.map', time.time()-start_time)
dataset = dataset.with_format(type='torch')
print("Set torch!")
dataset.save_to_disk(f"pile_{args.Index:0>2}.hf")
# %%
