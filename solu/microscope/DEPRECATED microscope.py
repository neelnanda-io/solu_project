# %%
from neel.imports import *

torch.set_grad_enabled(False)

MAX_ACT_STORE_DIR = Path("/workspace/solu_project/solu/microscope/max_act_stores")

# %%
@dataclass
class MicroscopeConfig:
    seed: int = 42
    batch_size: int = 10
    model_name: str = ""
    top_k: int = 20
    dtype: str = "fp32"
    device: str = "cuda"
    log_every: int = 50
    log: bool = True
    version: int = 4
    mode: str = "logit"


# %%
# Get myself data
# tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")
# code_data = datasets.load_dataset("NeelNanda/codeparrot_clean_subset_train", split="train")
# tokens_code = tokenize_and_concatenate(code_data, tokenizer, streaming=False, column_name="content")
# tokens_code.save_to_disk(f'/workspace/data/code_valid_tokens.hf')
# %%
# c4_urls = [f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:0>5}-of-01024.json.gz" for i in range(1000, 1008)]
# c4_data = load_dataset('json', data_files=c4_urls, split='train')
# tokens_c4 = tokenize_and_concatenate(c4_data, tokenizer, streaming=False, column_name="text")
# tokens_c4.save_to_disk(f'/workspace/data/c4_valid_tokens.hf')
# %%
# Make data loader


def get_c4_code(seed=0):
    tokens_code = datasets.load_from_disk(f"/workspace/data/code_valid_tokens.hf")
    tokens_c4 = datasets.load_from_disk(f"/workspace/data/c4_valid_tokens.hf")
    tokens_c4_code = datasets.concatenate_datasets([tokens_code, tokens_c4])
    tokens_c4_code = tokens_c4_code.with_format("torch")
    # Shuffle but with fixed seed - we want deterministic order, but also want to mix code + C4
    tokens_c4_code = tokens_c4_code.shuffle(seed=seed)
    return tokens_c4_code


def get_pile(seed=0):
    tokens_pile = datasets.load_from_disk("/workspace/solu_project/pile_29.hf")
    tokens_pile = tokens_pile.with_format("torch")
    # Shuffle but with fixed seed - we want deterministic order, but also want to distribute related texts
    tokens_pile = tokens_pile.shuffle(seed=seed)
    return tokens_pile


# %%
# dl = DataLoader(tokens_data, batch_size=mcfg.batch_size, shuffle=False, num_workers=8)
# # data_iter = iter(dl)
# for i in range(20):
#     toks = next(data_iter)
# offset = 190
# for j in range(10):
#     print(j)
#     print(tokenizer.decode(toks['tokens'][j, :50]))
#     print(tokenizer.decode(tokens_data[offset+j]['tokens'][:50]))
# %%
# Load model


# %%
# Define max act store
""" 
Brainstorm:
Needs a place to store max acts and text index
No need to keep them sorted
Each also maintains a minimum act - meh, improves constant but probs not worth it. 
Needs an update method, takes in a vector, compares to min act. Then there's a subupdate, which actually updates. Returns number of updates made
Needs a batch update which sorts, and then stops when num updates == 0
Needs a save method, which saves the two tensors to disk, plus a config file
Needs a load class method which does the alternate obvious thing
"""

DTYPE_DICT = {
    "fp32": torch.float32,
    "int64": torch.int64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class MaxActStore:
    def __init__(
        self,
        top_k,
        d_mlp,
        device="cuda",
        dtype="fp32",
        log_every=50,
        log=False,
        name="",
    ):
        self.top_k = top_k
        self.d_mlp = d_mlp
        self.device = device
        self.dtype = dtype
        self.max_acts = -torch.inf * torch.ones(
            (d_mlp, top_k), dtype=DTYPE_DICT[dtype], device=device
        )
        self.index = -torch.ones((d_mlp, top_k), dtype=torch.long, device=device)

        self.counter = 0
        self.total_updates = 0
        self.batches_seen = 0
        self.log_every = log_every
        self.start_time = time.time()
        self.log = log
        self.name = name

    def update(self, new_act, new_index):
        min_max_act, min_indices = self.max_acts.min(1)
        mask = min_max_act < new_act
        num_updates = mask.sum().item()
        self.max_acts[mask, min_indices[mask]] = new_act[mask]
        self.index[mask, min_indices[mask]] = new_index[mask]
        self.total_updates += num_updates
        return num_updates

    def batch_update(self, activations, text_indices=None):
        """
        activations: Shape [batch, d_mlp]
        text_indices: Shape [batch,]

        activations is the largest MLP activation, text_indices is the index of the text strings.

        Sorts the activations into descending order, then updates with each column until we stop needing to update
        """
        batch_size = activations.size(0)
        new_acts, sorted_indices = activations.sort(0, descending=True)
        if text_indices is None:
            text_indices = torch.arange(
                self.counter,
                self.counter + batch_size,
                device=self.device,
                dtype=torch.int64,
            )
        new_indices = text_indices[sorted_indices]
        new_updates = 0
        for i in range(batch_size):
            num_updates = self.update(new_acts[i], new_indices[i])
            new_updates += num_updates
            if num_updates == 0:
                break
        self.counter += batch_size
        self.batches_seen += 1
        if self.batches_seen % self.log_every == 0 and self.log:
            wandb.log(
                {
                    f"{self.name}_max_act": activations.max().item(),
                    f"{self.name}_new_updates": new_updates,
                    f"{self.name}_total_updates": self.total_updates,
                    f"{self.name}_steps_needed": i + 1,
                    f"{self.name}_batches_seen": self.batches_seen,
                    f"{self.name}_elapsed": time.time() - self.start_time,
                },
                step=self.counter,
            )
        return i

    def save(self, folder_name):
        path = MAX_ACT_STORE_DIR / folder_name
        path.mkdir(exist_ok=True)
        torch.save(self.max_acts, path / "max_acts.pth")
        torch.save(self.index, path / "index.pth")
        with open(path / "config.json", "w") as f:
            filt_dict = {
                k: v for k, v in self.__dict__.items() if k not in ["max_acts", "index"]
            }
            json.dump(filt_dict, f)

    @classmethod
    def load(self, folder_name):
        path = MAX_ACT_STORE_DIR / folder_name
        max_acts = torch.load(path / "max_acts.pth")
        index = torch.load(path / "index.pth")
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        mas = MaxActStore(config["top_k"], config["d_mlp"])
        for k, v in config.items():
            mas.__dict__[k] = v
        mas.max_acts = max_acts
        mas.index = index
        return mas

    def __repr__(self):
        return f"MaxActStore(top_k={self.top_k}, d_mlp={self.d_mlp}, counter={self.counter}, total_updates={self.total_updates}, device={self.device}, dtype={self.dtype}, log_every={self.log_every})\n Max Acts: {get_corner(self.max_acts)}\n Indices: {get_corner(self.index)}"


# %%
# Old code, has since been moved + substantially improved to scan_over_data.py

if False:
    with torch.autocast("cuda", torch.float32):
        for model_name in [
            "solu-1l-c4-code",
            "solu-2l-c4-code",
            "solu-3l-c4-code",
            "solu-4l-c4-code",
            "gelu-1l-c4-code",
            "gelu-2l-c4-code",
            "gelu-3l-c4-code",
            "gelu-4l-c4-code",
        ]:
            # for model_name in ['solu-2l-old', 'solu-4l-old', 'solu-6l-old', 'solu-8l-old', 'solu-10l-old']:
            print("Starting microscope for model_name:", model_name)
            mcfg = MicroscopeConfig(model_name=model_name)
            model = HookedTransformer.from_pretrained(mcfg.model_name)
            print(model.cfg)
            if model.cfg.tokenizer_name == "EleutherAI/gpt-neox-20b":
                print("Using Pile data")
                tokens_data = get_pile()
                tokens_name = "text"
            elif model.cfg.tokenizer_name == "NeelNanda/gpt-neox-tokenizer-digits":
                print("Using C4 + Code data")
                tokens_data = get_c4_code()
                tokens_name = "tokens"
            else:
                raise ValueError(f"Invalid Tokenizer! {model.cfg.tokenizer_name}")

            stores = [
                MaxActStore(
                    mcfg.top_k,
                    model.cfg.d_mlp,
                    device=mcfg.device,
                    dtype=mcfg.dtype,
                    log_every=mcfg.log_every,
                    log=mcfg.log,
                    name=str(i),
                )
                for i in range(model.cfg.n_layers)
            ]

            if mcfg.mode == "max_act":
                # Add the hooks to hook_mid
                def store_max_act(mid_act, hook, index):
                    # Take max over position dimension
                    max_act = mid_act.max(-2).values
                    # Update the store
                    stores[index].batch_update(max_act)

                for index in range(model.cfg.n_layers):
                    model.blocks[index].mlp.hook_mid.add_hook(
                        partial(store_max_act, index=index)
                    )
            else:
                # Add the hooks to hook_mid
                tokens = None

                def store_logit_attr(post_act, hook, index):
                    # Take max over position dimension
                    logit_attr = einsum(
                        "batch pos d_mlp, d_mlp d_model, d_model batch pos -> batch pos d_mlp",
                        post_act,
                        model.blocks[index].mlp.W_out,
                        model.unembed.W_U[:, tokens],
                    )
                    # Update the store
                    stores[index].batch_update(logit_attr.max(-2).values)

                for index in range(model.cfg.n_layers):
                    model.blocks[index].mlp.hook_post.add_hook(
                        partial(store_logit_attr, index=index)
                    )

            # Run across entire dataset
            # try:
            run_name = f"{mcfg.model_name}_v{mcfg.version}"
            if mcfg.log:
                wandb.init(project="max_act_store", config=mcfg, name=run_name)
            print("Starting training!")
            test = tokens_data[0 : 0 + mcfg.batch_size][tokens_name].to(mcfg.device)
            print(get_corner(test))
            print(model.tokenizer.batch_decode(test)[0][:100])
            for batch in tqdm.tqdm(
                range(0, min(len(tokens_data), 2 * 10**6), mcfg.batch_size)
            ):
                tokens = tokens_data[batch : batch + mcfg.batch_size][tokens_name].to(
                    mcfg.device
                )
                model(tokens, return_type=None)
                if (batch // mcfg.batch_size) % 1000 == 0:
                    print(f"Finished {batch} batches")
            # except Exception as e:
            #     print("An error happening, moving to saving:", e)
            print(f"Saving stores to {mcfg.model_name}")
            for c, store in enumerate(stores):
                store.save(f"{run_name}L{c}")
            wandb.finish()
