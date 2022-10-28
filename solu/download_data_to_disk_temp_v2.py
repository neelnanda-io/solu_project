# %%
from neel.imports import *
n = 160

c4_urls = [f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:0>5}-of-01024.json.gz" for i in range(n)]

dataset = load_dataset('json', data_files=c4_urls, split='train')
# dataset.save_to_disk(Path.home()/f'data/c4_train_{n}_text.hf')
print(dataset)
# # %%
from easy_transformer.utils import tokenize_and_concatenate
tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")
# %%
tokens_cc = tokenize_and_concatenate(dataset, tokenizer, streaming=False)

tokens_cc.save_to_disk(Path.home()/f'data/c4_train_{n}_tokens.hf')

# print(tokens_cc2)

# %%
# mode = "train"
# code_data = load_dataset(f"codeparrot/codeparrot-{mode}-v2-near-dedup", split="tr1ain")

# code_data = code_data.train_test_split(0.05)
# # code_data.save_to_disk(Path.home()/f"data/codeparrot_{mode}_split.hf")
# code_data_train = code_data['train']

# tokens_code = tokenize_and_concatenate(code_data_train, tokenizer, streaming=False, column_name="content")
# tokens_code.save_to_disk(Path.home()/f'data/codeparrot_{mode}_tokens.hf')
# %%
def replace_zero(tokens):
    tokens = tokens['tokens']
    mask = torch.zeros_like(tokens, dtype=bool)
    mask[:, 1:] = tokens[:, 1:]==1
    tokens = torch.where(mask, torch.tensor(0), tokens)
    return {"tokens":tokens}
# %%
# data = datasets.load_from_disk(Path.home()/"data/codeparrot_train_tokens.hf")
# data = data.with_format("torch")
# data = data.map(replace_zero, batched=True, batch_size=10000)
# data.save_to_disk(Path.home()/"data/codeparrot_train_tokens_v2.hf")
# %%
data = datasets.load_from_disk(Path.home()/"data/c4_train_160_tokens.hf")
data = data.with_format("torch")
data = data.map(replace_zero, batched=True, batch_size=10000)
data.save_to_disk(Path.home()/"data/c4_train_tokens_v2.hf")
# %%
