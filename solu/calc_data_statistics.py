# %%
from neel.imports import *

cfg = dict(
    data_name="code",
    max_tokens=int(1e12),
    batch_size=1000000,
)
cfg = sutils.arg_parse_update_cfg(cfg)

if cfg["data_name"] == "pile":
    d_vocab = 50278
elif cfg["data_name"] == "openwebtext":
    d_vocab = 50257
else:
    d_vocab = 48262
dataset = sutils.get_dataset(cfg["data_name"])
cfg["max_tokens"] = min(cfg["max_tokens"], len(dataset))

unigram = torch.zeros(d_vocab, dtype=torch.int32)
bigram = torch.zeros(d_vocab**2, dtype=torch.int32)

for index in tqdm.tqdm(range(0, cfg["max_tokens"], cfg["batch_size"])):
    if cfg["data_name"] == "openwebtext":
        big_tensor = dataset[index : index + cfg["batch_size"]]["tokens"].to(torch.int64)
    else:
        big_tensor = dataset[index : index + cfg["batch_size"]]["tokens"].to(
            torch.int64
        )
    unigram += torch.bincount(big_tensor.flatten(), minlength=len(unigram)).int()
    bigram_tensor = big_tensor[:, :-1] + d_vocab * big_tensor[:, 1:]
    bigram += torch.bincount(bigram_tensor.flatten(), minlength=len(bigram)).int()

dir = Path(f"/workspace/data/stats/{cfg['data_name']}")
dir.mkdir(exist_ok=True, parents=True)
torch.save(unigram, dir / "unigram.pth")
torch.save(bigram.to_sparse(), dir / "bigram.pth")
# %%
