import os
import time

"""
os.system("/opt/conda/bin/python /workspace/solu_project/solu/microscope/scan_over_data.py --use_pred_log_probs --use_max_neuron_act --use_neuron_logit_attr --use_head_logit_attr --use_activation_stats --model_name solu-1l --batch_size 2 --version 0 --overwrite --data_name code")
raise ValueError
"""
model_names = [
    "solu-6l",
    "solu-8l",
    "solu-10l",
    "solu-12l",
]
data_names = ["code", "c4"]
for data_name in data_names:
    for model_name in model_names:
        # os.system(f"/opt/conda/bin/python /workspace/solu_project/solu/microscope/scan_over_data.py --use_activation_stats --model_name {model_name} --data_name {data_name}")
        os.system(
            f"/opt/conda/bin/python /workspace/solu_project/solu/microscope/scan_over_data.py --use_pred_log_probs --use_max_neuron_act --use_neuron_logit_attr --use_head_logit_attr --use_activation_stats --version 2 --model_name {model_name} --data_name {data_name}"
        )
        # Gives time for keyboard interrupt
        time.sleep(3)
