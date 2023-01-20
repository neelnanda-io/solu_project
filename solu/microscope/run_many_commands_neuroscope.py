import os
import time
import transformer_lens.loading_from_pretrained as loading
import solu.utils as sutils
"""
os.system("/opt/conda/bin/python /workspace/solu_project/solu/microscope/scan_over_data.py --use_pred_log_probs --use_max_neuron_act --use_neuron_logit_attr --use_head_logit_attr --use_activation_stats --model_name solu-1l --batch_size 2 --version 0 --overwrite --data_name code")
raise ValueError
"""
params = {"PARITY": -1, "num_parallel_runs": 2}
params = sutils.arg_parse_update_cfg(params)
num_parallel_runs = params["num_parallel_runs"]
assert params["PARITY"]!=-1
PARITY = params["PARITY"]
print("PARITY", PARITY)
print("num_parallel_runs", num_parallel_runs)
model_names = [
    # "gelu-2l",
    # "gelu-3l",
    # "gelu-4l",
    # "solu-1l",
    # "solu-2l",
    # "gelu-1l",
    # "gelu-2l",
    # "solu-12l",
    # "gpt2-small",
    # "solu-3l",
    # "gelu-3l",
    # "solu-4l",
    # "gelu-4l",
    # "solu-10l-pile",
    # "gpt2-medium",
    # "gpt2-large",
    "gpt2-xl",
    # "solu-10l",
    "solu-8l",
    "solu-6l",
    # "solu-1l-pile",
    # "solu-2l-pile",
    # "solu-4l-pile",
    # "solu-6l-pile",
    # "solu-8l-pile",
    # "solu-12l-pile",
    # "gpt-neo-small",
    # "pythia-19m-deduped",
    # "pythia-125m-deduped",
    # "pythia-350m-deduped",
]
# model_names = [
#     "solu-1l-pile",
#     "solu-2l-pile",
#     "solu-4l-pile",
#     "solu-6l-pile",
#     "solu-8l-pile",
#     "solu-10l-pile",
# ]
command_index = 0
for model_name in model_names:
    cfg = loading.get_pretrained_model_config(model_name)
    for layer in range(cfg.n_layers):
        if model_name == "gpt2-large" and layer < 25:
            continue
        if model_name == "gpt2-xl" and layer < 28:
            continue
        command = f"/opt/conda/bin/python /workspace/solu_project/solu/microscope/make_leoxscope_page.py --model_name {model_name} --layer {layer}"
        if command_index % num_parallel_runs == PARITY:
        # if True:
            print("Running command:", command)
            os.system(command)
        else:
            print("Skipping command:", command)

        command_index += 1
        # Gives time for keyboard interrupt
        time.sleep(3)
