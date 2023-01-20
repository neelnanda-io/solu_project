import os
import time

"""
os.system("/opt/conda/bin/python /workspace/solu_project/solu/microscope/scan_over_data.py --use_pred_log_probs --use_max_neuron_act --use_neuron_logit_attr --use_head_logit_attr --use_activation_stats --model_name solu-1l --batch_size 2 --version 0 --overwrite --data_name code")
raise ValueError
"""
model_names = [
    # "solu-1l-pile",
    # "solu-2l-pile",
    "solu-6l-pile",
    "solu-12l-pile",
    "pythia-19m-deduped",
    "pythia-125m-deduped",
    "gpt-neo-small",
    "solu-4l-pile",
    "solu-8l-pile",
    "solu-10l-pile",
    "pythia-350m-deduped",
]
data_name = "pile"
for model_name in model_names:
    os.system(
        f"/opt/conda/bin/python /workspace/solu_project/solu/microscope/scan_over_data.py --use_max_neuron_act --version 2 --model_name {model_name} --data_name {data_name} --use_activation_stats"
    )
    # Gives time for keyboard interrupt
    time.sleep(3)
