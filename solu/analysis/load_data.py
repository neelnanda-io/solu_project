# %%
import time
import datasets
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import einops
start_time = time.time()
# if cfg['shuffled_data']:
# randperm = np.random.permutation(28)
# print('Permutation of PILE URLs', randperm)
# pile_urls = [f"https://mystic.the-eye.eu/public/AI/pile/train/{i:0>2}.jsonl.zst" for i in randperm]
dataset = load_dataset('json', data_files="https://mystic.the-eye.eu/public/AI/pile/train/29.jsonl.zst", split='train', cache_dir='cache')
print(dataset[0])
    # else:
    #     dataset = load_dataset(cfg['dataset_name'], streaming=True, split='train')
print('Loaded!', time.time()-start_time)
start_time = time.time()
try:
    dataset = dataset.remove_columns('meta')
except:
    print('Meta not in dataset')
print(dataset[0])
print("Formatting")
start_time = time.time()

print(dataset[0])
print('dataset.set_format', time.time()-start_time)
start_time = time.time()
# dataset = dataset.shuffle(seed=cfg['seed'], buffer_size=30000)
print('dataset.shuffle', time.time()-start_time)
start_time = time.time()
# train_data_loader = DataLoader(dataset, batch_size=cfg['batch_size'])
# %%
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
pad_token = '<PAD>'
tokenizer.add_special_tokens({'pad_token':pad_token})
print(tokenizer)

seq_len = 1024
# %%
def tokenize(examples):
    # print("Hello!")
    start_time = time.time()
    texts = examples['text']
    # print(texts)
    full_text = tokenizer.eos_token.join(texts)
    div = 20
    length = len(full_text)//div
    text_list = [full_text[i*length:(i+1)*length] for i in range(div)]
    tokens = tokenizer(text_list, return_tensors='np', padding=True)['input_ids'].flatten()
    tokens = tokens[tokens!=tokenizer.pad_token_id]
    # print(len(text_list), len(text_list[0]))
    # print(tokens.shape)
    n = len(tokens)
    curr_batch_size = n//(seq_len-1)
    tokens = tokens[:(seq_len-1)*curr_batch_size]
    tokens = einops.rearrange(tokens, '(batch_size seq) -> batch_size seq', batch_size=curr_batch_size, seq=seq_len-1)
    prefix = np.ones((curr_batch_size, 1), dtype=np.int64)*tokenizer.bos_token_id
    # print(tokens.shape, n, curr_batch_size, seq_len)
    return {'text': np.concatenate([prefix, tokens], axis=1)}# tiny_owt_orig_2 = load_dataset('stas/openwebtext-10k', cache_dir='./cache', split='train', download_config=datasets.DownloadConfig(resume_download=True, num_proc=4))

# print('Loaded!', time.time()-start_time)
start_time = time.time()
# dataset = dataset.map(tokenize, batched=False, batch_size=10)
dataset = dataset.map(tokenize, batched=True, num_proc=20)
print('dataset.map', time.time()-start_time)
dataset = dataset.with_format(type='torch')
print("Set torch!")
dataset.save_to_disk("pile_29.hf")
# %%
