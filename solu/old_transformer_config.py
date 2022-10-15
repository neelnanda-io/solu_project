from collections import OrderedDict
NUM_DEVICES = 8

DEFAULT_CONFIG = OrderedDict({
        "d_model": -1,
        "d_head": 64,
        "d_mlp": -1,
        "d_vocab": 50278,
        "n_params": -1,
        "n_layers": -1, # Not used, just for logging
        "n_heads": -1,
        "n_ctx": 1024,
        "seed": -1,
        "debug": True,
        "batch_size": 16,
        "batches_per_step": 1,
        "lr": 1e-3,
        "max_tokens": 22 * 10**9,
        "take_checkpoints": True,
        "checkpoint_scale": 1,
        "lr_schedule": "cosine_warmup",
        "warmup_tokens": 25*10**7,
        "normalization": "LN",
        "beta1": 0.9,
        "beta2": 0.99,
        "ln_eps": 1e-5,
        "weight_decay": 0.01,
        "dataset_name": "the_pile",
        "grad_norm_clip": 1.0,
        "act_fn": "SoLU",
        "n_devices": NUM_DEVICES,
        "train_loss_ewma_beta": 0.99,
        "dtype": "bf16",
    })
def create_config(n_layers, **parsed_args):
    cfg = dict(DEFAULT_CONFIG)
    cfg["n_layers"] = n_layers
    cfg.update(parsed_args)

    if cfg['d_model']==-1:
        cfg['d_model'] = 128 * cfg['n_layers']
    if cfg['d_mlp']==-1:
        cfg["d_mlp"] = 4 * cfg['n_layers']
    if cfg["n_heads"]==-1:
        if (cfg['d_model'] % cfg['d_head']) != 0:
            logging.warning(f"d_model {cfg['d_model']} and d_head {cfg['d_head']} do not divide evenly")
        cfg["n_heads"] = cfg["d_model"] // cfg["d_head"]
    cfg["n_params"] = 12 * cfg["n_layers"] * cfg["d_model"] ** 2


    cfg['n_devices'] = NUM_DEVICES
    cfg['total_batch_size'] = cfg['batch_size'] * cfg['n_devices']
    cfg["tokens_per_step"] = cfg["total_batch_size"] * cfg["n_ctx"] * cfg["batches_per_step"]
    cfg["max_steps"] = cfg["max_tokens"] // cfg["tokens_per_step"]
    cfg["warmup_steps"] = cfg["warmup_tokens"] // cfg["tokens_per_step"]
    
    cfg["version"] = max(solu_get_prev_versions()[0])+1

    if cfg["debug"]:
        cfg["max_steps"] = 25
    
    if cfg["seed"]==-1:
        cfg["seed"] = cfg["version"]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    set_seed(cfg["seed"])
    return cfg
cfg = create_config(12)
rprint(cfg)