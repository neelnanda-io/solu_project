# %%
#! Imports!

from neel.imports import *

#! Fun Imports
from accelerate import Accelerator
from accelerate.utils import set_seed, write_basic_config
from accelerate import notebook_launcher
from easy_transformer.utils import tokenize_and_concatenate
import argparse

#! SoLU Imports
from solu.old_transformer import Transformer
from solu.utils import solu_get_prev_versions
# %%
# %%

#! Test main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_layers", type=int)
    for key, value in DEFAULT_CONFIG.items():
        if key!="n_layers":
            parser.add_argument(f"--{key}", type=type(value), default=value)
    
    if IN_IPYTHON:
        args = parser.parse_args(["1"])
    else:
        args = parser.parse_args()
    
    cfg = create_config(**vars(args))

    print(f"Config for {cfg['n_layers']}L v{cfg['version']} with {cfg['n_params']/1e6:.2f}M params")
    for key in (cfg.keys()):
        print(f"{key}: {cfg[key]}")
    print(f"Config for {cfg['n_layers']}L v{cfg['version']} with {cfg['n_params']/1e6:.2f}M params")
# %%
#! Config

CHECKPOINT_DIR = Path("/workspace/solu_project/solu_checkpoints/")
NUM_TRAIN_DATA_FILES = 1
TRAIN_DATA_FILES = [f'/workspace/data/pile_{i:0<2}.hf' for i in range(NUM_TRAIN_DATA_FILES)]
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_layers", type=int)



# %%
#! Helper Functions
def get_corner(tensor, n=2):
    # Prints the top left corner of the tensor
    if len(tensor.shape) == 0:
        return tensor
    elif len(tensor.shape) == 1:
        return tensor[:n]
    elif len(tensor.shape) == 2:
        return tensor[:n, :n]
    elif len(tensor.shape) == 3:
        return tensor[:n, :n, :n]
    elif len(tensor.shape) == 4:
        return tensor[:n, :n, :n, :n]
    elif len(tensor.shape) == 5:
        return tensor[:n, :n, :n, :n, :n]
    elif len(tensor.shape) == 6:
        return tensor[:n, :n, :n, :n, :n, :n]
    else:
        # I never need tensors of rank > 6
        raise ValueError(f"Tensor of shape {tensor.shape} is too big")

def to_numpy(tensor, flat=False):
    if (type(tensor) != torch.Tensor) and (
        type(tensor) != torch.nn.parameter.Parameter
    ):
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()

def loss_fn(logits, batch):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, batch[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()

def init_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    pad_token = "<PAD>"
    tokenizer.add_special_tokens({"pad_token": pad_token})
    return tokenizer

#! Data

def create_dataset(cfg):
    data = datasets.concatenate_datasets([datasets.load_from_disk(file_name)
        for file_name in TRAIN_DATA_FILES])
    print(data)
    data = data.with_format("torch")
    data = data.shuffle(seed=cfg['seed'])

    loader = torch.utils.data.DataLoader(data, batch_size=cfg["batch_size"], shuffle=False, num_workers=8)
    return loader

#! Checkpointing

class SaveSchedule:
    """
    Decides when to save the model, has a hard-coded schedule of 162 checkpoints, which can be (approx) scaled up or down in frequency with the `scale` parameter.
    """
    def __init__(self, max_tokens, tokens_per_step, scale=1, custom_schedule=None, show_plot=False):
        self.scale = scale
        if custom_schedule is None:
            self.schedule = np.concatenate(
                [
                    self.normalised_range(1, 10) * 1e-3,
                    self.normalised_range(2, 20) * 1e-2,
                    self.normalised_range(5, 50) * 1e-1,
                    self.normalised_range(10, 100),
                ]
            )
        else:
            self.schedule = custom_schedule
        self.max_tokens = max_tokens
        self.tokens_per_step = tokens_per_step
        self.counter = 0
        self.next_save_point = 0
        if show_plot:
            # Plots the save schedule over training
            px.line(
                self.schedule * max_tokens,
                log_y=True,
                title="Save Schedule",
                labels={"y": "Tokens", "x": "Checkpoint Index"},
            ).show()

    def step(self):
        value = self.counter * self.tokens_per_step / self.max_tokens
        threshold = self.schedule[self.next_save_point]
        if value >= threshold:
            self.next_save_point += 1
            self.counter += 1
            return True
        else:
            self.counter += 1
            return False
    
    def normalised_range(self, start, stop):
        stop = round(stop * self.scale)
        start = round(start * self.scale)
        return np.arange(start, stop)/stop

schedule = SaveSchedule(cfg["max_tokens"], cfg["tokens_per_step"], scale=3.2, show_plot=True)

# %%
#! Save Model
def save_checkpoint(model, optimizer, scheduler, cfg, step, folder_name):
    if cfg["take_checkpoints"]:
        model_name = f"SoLU_{folder_name}_{step:0<6}.pth"
        torch.save(optimizer.state_dict(), CHECKPOINT_DIR/folder_name/f"SoLU_{folder_name}_optimizer_checkpoint.pth")
        torch.save(scheduler.state_dict(), CHECKPOINT_DIR/folder_name/f"SoLU_{folder_name}_scheduler_checkpoint.pth")
        torch.save(model.state_dict(), CHECKPOINT_DIR/folder_name/model_name)
        print(f"Saved model to {model_name}")
# %%



def old_main(mixed_precision="bf16", seed: int = 42):
    set_seed(seed)

    accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=2)

    cfg = create_cfg()
    assert cfg["batches_per_step"] == accelerator.gradient_accumulation_steps

    data_iter = create_dataset(cfg, accelerator)

    accelerator.print("initialized accelerator")

    model_name = f'SoLU_{cfg["n_layers"]}L_v{cfg["version"]}'

    if accelerator.is_main_process:
        wandb.init(project="solu", entity="mechanistic-interpretability", config=cfg)

        torch.save(cfg, model_name + "_config.pth")
        wandb.save(model_name + "_config.pth")

    tokenizer = init_tokenizer(accelerator)
    # device = accelerator.device

    if cfg['use_ET']:
        from easy_transformer.EasyTransformerConfig import EasyTransformerConfig
        from easy_transformer import EasyTransformer
        from rich import print as rprint
        from dataclasses import dataclass

        # %%
        from IPython import get_ipython
        try:
            ipython = get_ipython()
            # Code to automatically update the EasyTransformer code as its edited without restarting the kernel
            ipython.magic("load_ext autoreload")
            ipython.magic("autoreload 2")
            import plotly.io as pio
            pio.renderers.default = "vscode"
            IS_IPYTHON = True
        except:
            IS_IPYTHON = False


        # %%
        @dataclass
        class TrainingConfig:
            apply_anthropic_hyper_params: bool
        #     lr: float
        #     batch_size: int

        #     seed: 12345
        #     batches_per_step: int = 1



            model_cfg: EasyTransformerConfig = None

            @classmethod
            def from_dict(cls, **cfg_dict):
                model_config_keys = EasyTransformerConfig.__dataclass_fields__.keys()

                model_cfg_dict = {k:v for k, v in cfg_dict.items() if k in model_config_keys}
                training_cfg_dict = {k:v for k, v in cfg_dict.items() if k not in model_config_keys}

                if training_cfg_dict['apply_anthropic_hyper_params']:
                    n_layers = model_cfg_dict['n_layers']
                    model_cfg_dict['d_model'] = n_layers * 128
                    model_cfg_dict['d_mlp'] = 4 * model_cfg_dict['d_model']
                    model_cfg_dict['d_head'] = 64
                    assert model_cfg_dict['d_model'] % model_cfg_dict['d_head'] == 0, f"d_head: {model_cfg_dict['d_head']} is not a divisor of d_model: {model_cfg_dict['d_model']}"
                    model_cfg_dict['n_heads'] = model_cfg_dict['d_model']//model_cfg_dict['d_head']

                model_cfg = EasyTransformerConfig.from_dict(model_cfg_dict)

                # rprint(training_cfg_dict)
                # rprint(model_cfg_dict)
                return cls(model_cfg = model_cfg, **training_cfg_dict)

        config = TrainingConfig.from_dict(
            n_layers = 4,
            apply_anthropic_hyper_params = True,
            act_fn='solu_ln',
            tokenizer_name = "EleutherAI/gpt-neox-20b",
        #     device='cpu',
        #     lr=1e-4,
            n_ctx=1024,
        )
        # rprint(config)

        # %%
        model = EasyTransformer.from_config(config.model_cfg)
        accelerator.print(model.cfg)
        # rprint(model)
    else:
        model = Transformer(cfg, tokenizer)

    model.to(accelerator.device)
    if cfg["use_bfloat16"]:
        model.to(torch.bfloat16)
    elif cfg["use_float16"]:
        model.to(torch.float16)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=cfg["betas"],
        weight_decay=cfg["weight_decay"],
    )
    if cfg["lr_schedule"] is not None:

        def lr_schedule(step):
            if step < cfg["warmup_steps"]:
                return (1e-7 + (cfg["lr"] - 1e-7) * step / cfg["warmup_steps"]) / cfg[
                    "lr"
                ]
            else:
                return 0.55 + 0.9 * 0.5 * np.cos(
                    np.pi
                    * (step - cfg["warmup_steps"])
                    / (cfg["max_steps"] - cfg["warmup_steps"])
                )

        param_groups = {"decay": [], "no_decay": []}
        for name, param in model.named_parameters():
            accelerator.print(name)
            accelerator.print(param.dtype)
            if "W_" in name and name not in ["W_E", "W_U"]:
                param_groups["decay"].append(param)
            else:
                param_groups["no_decay"].append(param)
        optim_groups = [
            {"params": param_groups["decay"], "weight_decay": cfg["weight_decay"]},
            {"params": param_groups["no_decay"], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=cfg["lr"])
        accelerator.print(optimizer)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    if accelerator.is_main_process:
        schedule = SaveSchedule(
            cfg["max_tokens"],
            cfg["tokens_per_step"],
        )

    model, optimizer, data_iter, scheduler = accelerator.prepare(model, optimizer, data_iter, scheduler)

    accelerator.print(cfg)
    # DataLoader(full_owt_test['text'], batch_size=cfg['batch_size'], shuffle=False, pin_memory=False)
    accelerator.print("Training begins!")
    losses = []
    loss_ewmas = []
    step = 0
    start_time = time.time()
    loss_ewma = 9
    # loss_beta = 0.95
    total_tokens = 0
    # running_loss = 0
    prev_time = time.time()
    epoch = 0
    # for epoch in range(100):
    running_loss = torch.tensor(0., requires_grad=False).to(accelerator.device)
    for c, batch in tqdm.tqdm(enumerate(data_iter)):
        with accelerator.accumulate(model):
            batch = batch["text"]
            if c<=3:
                accelerator.print(accelerator.is_main_process, "Batch shape:", batch.shape)
            
            # batch = batch.cuda()
            with accelerator.autocast():
                if cfg['use_ET']:
                    loss = model(batch, return_type='loss')
                else:
                    loss = model(batch)

            # loss = loss / accelerator.num_processes
            accelerator.backward(loss)

            running_loss += loss.detach()
            # batch_tokens = torch.tensor(batch.numel(), device=accelerator.device)
            # dist.all_reduce(batch_tokens, op=dist.ReduceOp.SUM)
            total_tokens += batch.numel()*cfg["n_devices"]
            if (c + 1) % cfg["batches_per_step"] == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg["grad_norm_clip"])
                optimizer.step()
                if cfg["lr_schedule"] is not None:
                    scheduler.step()
                    if accelerator.is_main_process:
                        wandb.log({"scheduled_lr": scheduler.get_last_lr()[0]}, step=step)
                optimizer.zero_grad()
                if (
                    accelerator.is_main_process
                    and schedule.step()
                    and cfg["use_checkpoint_schedule"]
                ):
                    accelerator.print(
                        f"Saved the model! Step: {step}. Frac of way through training: {schedule.schedule[schedule.next_save_point-1]*accelerator.num_processes}"
                    )
                    if not cfg["debug"]:
                        if cfg["save_checkpoints_to_bfloat16"]:
                            save_to_bfloat16(model, f"{model_name}_{step:0>6}.pth")
                        else:
                            torch.save(model.state_dict(), f"{model_name}_{step:0>6}.pth")
                        torch.save(
                            optimizer.state_dict(), f"{model_name}_opt_checkpoint.pth"
                        )
                        if cfg["lr_schedule"] is not None:
                            torch.save(
                                scheduler.state_dict(),
                                f"{model_name}_scheduler_checkpoint.pth",
                            )
                        wandb.save(f"{model_name}_{step:0>6}.pth")
                
                # dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
                # running_loss /= accelerator.num_processes
                # print(accelerator.local_process_index, running_loss)
                # losses.append(running_loss)

                loss_ewma = loss_ewma * cfg["train_loss_ewma_beta"] + running_loss * (
                    1 - cfg["train_loss_ewma_beta"]
                )
                loss_ewmas.append(loss_ewma)
                if accelerator.is_main_process:
                    wandb.log(
                        {
                            "loss": 9.2,
                            "loss_ewma": 8.5,
                            "elapsed": time.time() - start_time,
                            "total_tokens": total_tokens,
                            "c": c,
                        },
                        step=step,
                    )
                # accelerator.print("Just logged")
                # accelerator.print(
                #     {
                #         "loss": loss.item(),
                #         "loss_ewma": loss_ewma,
                #         "elapsed": time.time() - start_time,
                #         "total_tokens": total_tokens,
                #         "c": c,
                #     }
                # )
                running_loss = 0
                # if step % 30 == 0:
                #     accelerator.print(c, step, total_tokens, losses[-1], loss_ewmas[-1])
                step += 1
                # if step >= cfg["max_steps"]:
                #     break
                if total_tokens > cfg["max_tokens"]:
                    break
            if c <= 12 and epoch == 0:
                # cuda_memory()
                accelerator.print("Early iteration complete!", c, time.time() - prev_time)
                prev_time = time.time()
            del loss
        # print(batch.shape, logits.shape, running_loss, loss, step, total_tokens)
        # if not cfg['debug_overfit']:
        #     break

    accelerator.print(f"Finished training! Train Loss EWMA: {loss_ewma}")

    if not cfg["debug"] and accelerator.is_main_process:
        torch.save(model.state_dict(), f"{model_name}_final.pth")
        wandb.save(f"{model_name}_final.pth")
        wandb.finish()


if __name__ == "__main__":
    main()

# %%
