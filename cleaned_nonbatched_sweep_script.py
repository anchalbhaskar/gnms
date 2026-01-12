# %%
import os, sys

repo_root = "/Users/anchalbhaskar/Desktop/CBU Work/gnms"
src_dir = os.path.join(repo_root, "src")  # or just repo_root if gnm is directly there

if src_dir in sys.path:
    sys.path.remove(src_dir)
sys.path.insert(0, src_dir)

sys.modules.pop("gnm", None)

import gnm
import gnm.defaults as defaults

print("gnm module:", gnm.__file__)
print("defaults module:", defaults.__file__)

# %%
import gnm
import gnm.defaults as defaults
print("gnm module:", gnm.__file__)
print("defaults module:", defaults.__file__)

# %%
from gnm.fitting import experiment_saving
from gnm.fitting.experiment_dataclasses import Experiment
from gnm import defaults, fitting, generative_rules, weight_criteria
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
distance_matrix = defaults.get_distance_matrix(device=DEVICE)

# %%
import numpy as np

file_path = "/Users/anchalbhaskar/Desktop/CBU Work/gnms/weighted_connectivity.npy"
weighted_connectivity = np.load(file_path)

print(weighted_connectivity.shape)

# %%
print("Original shape:", weighted_connectivity.shape)

# (90, 90, 446) → (446, 90, 90)
all_networks_np = np.transpose(weighted_connectivity, (2, 0, 1))
print("After transpose:", all_networks_np.shape)   # (446, 90, 90)

# 0 stays 0, anything > 0 becomes 1
all_binary_networks_np = (all_networks_np > 0).astype(np.float32)

print("Binary shape:", all_binary_networks_np.shape)  # (446, 90, 90)

all_binary_networks = torch.tensor(all_binary_networks_np, device=DEVICE)

S, N, _ = all_binary_networks.shape
print("Num subjects:", S)  #  446
print("Num nodes:", N)     # 90


# %%
num_connections = int(all_binary_networks.sum().item() / (446))
print(f"The binary consensus network contains {num_connections} connections.")

# %%
num_simulations = 10
file_path = "/Users/anchalbhaskar/Desktop/CBU Work/gnms/seed.npy"
seed = np.load(file_path)
seed_new = np.broadcast_to(seed, (num_simulations,seed.shape[0], seed.shape[1]))
tensor_seed = torch.tensor(seed_new, device=DEVICE)

# %%
eta_values = torch.Tensor(torch.linspace(-5, 0, 25))
gamma_values = torch.Tensor(torch.linspace(0, 1, 25))

# %%
binary_sweep_parameters = fitting.BinarySweepParameters(
    eta = eta_values,
    gamma = gamma_values,
    lambdah = torch.Tensor([0.0]),
    distance_relationship_type = ["powerlaw"],
    preferential_relationship_type = ["powerlaw"],
    heterochronicity_relationship_type = ["powerlaw"],
    generative_rule = [generative_rules.MatchingIndex()],
    num_iterations = [num_connections],
)

#seed set
sweep_config = fitting.SweepConfig(
    binary_sweep_parameters = binary_sweep_parameters,
    # weighted_sweep_parameters = weighted_sweep_parameters,
    num_simulations = num_simulations,
    distance_matrix = [distance_matrix], 
    seed_weight_matrix = tensor_seed   
)


# %%
from gnm.fitting.experiment_saving import ExperimentEvaluation
from gnm import evaluation

criteria = [ evaluation.ClusteringKS(), evaluation.DegreeKS(), evaluation.EdgeLengthKS(distance_matrix), evaluation.BetweennessKS() ]
energy = evaluation.MaxCriteria( criteria )
binary_evaluations = [energy]

# Run the full sweep
experiments = fitting.perform_sweep(sweep_config=sweep_config, 
                                binary_evaluations=binary_evaluations, 
                                real_binary_matrices= all_binary_networks,
                                # weighted_evaluations=weighted_evaluations,
                                save_model = False,
                                save_run_history = False,
                                verbose=True
)


# %%
eval = ExperimentEvaluation('./testingfile')
eval.save_experiments(experiments)

# %%
import json
import pandas as pd
import numpy as np
import sys
import os
def get_best_params_per_participant_multirun(json_file_path):
    """
    Reads a GNM index JSON file.
    Assumes 'per_connectome_binary_evals' contains keys representing runs (0, 1, ...).
    Inside each run, there is a list of values for participants.
    1. Finds the best parameter configuration (lowest energy) for each (participant, run) pair.
    2. Aggregates across runs for each participant to calculate Mean and STD of the parameters and the minimal energy.
    """
    print(f"Reading file: {json_file_path}")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    experiment_configs = data.get('experiment_configs', {})
    print(f"Found {len(experiment_configs)} configurations to process.")
    # Dictionary to store the best result for each (participant_idx, run_idx)
    # Key: (participant_idx, run_idx)
    # Value: {'min_val': float, 'params': dict}
    best_results_per_run = {}
    # Parameters to track
    param_keys = ['eta', 'gamma', 'lambdah', 'alpha']
    for config_name, config_data in experiment_configs.items():
        # Extract parameters for this config
        current_params = {}
        for key in param_keys:
            if key in config_data:
                current_params[key] = config_data[key]
        # 'per_connectome_binary_evals' -> keys are run indices (as strings), values are lists of energies per participant
        evals_container = config_data.get('per_connectome_binary_evals', {})
        for run_key, eval_list in evals_container.items():
            if not isinstance(eval_list, list):
                continue
            try:
                run_idx = int(run_key)
            except ValueError:
                continue
            # Iterate through participants in this run
            for p_idx, val in enumerate(eval_list):
                key = (p_idx, run_idx)
                # Initialize if not present
                if key not in best_results_per_run:
                    best_results_per_run[key] = {
                        'min_val': float('inf'),
                        'params': {}
                    }
                # Check for new minimum
                if val < best_results_per_run[key]['min_val']:
                    best_results_per_run[key]['min_val'] = val
                    best_results_per_run[key]['params'] = current_params.copy()
    # Now aggregate across runs for each participant
    # Organize data by participant_index
    participant_data = {}
    for (p_idx, run_idx), res in best_results_per_run.items():
        if res['min_val'] == float('inf'):
            continue
        if p_idx not in participant_data:
            participant_data[p_idx] = {
                'min_energy': [],
                'eta': [],
                'gamma': [],
                'lambdah': [],
                'alpha': [] # Alpha might be empty if not present, handle carefully
            }
        participant_data[p_idx]['min_energy'].append(res['min_val'])
        for p_key in param_keys:
            # Only append if the parameter exists in the best config found
            if p_key in res['params']:
                participant_data[p_idx][p_key].append(res['params'][p_key])
    # Create final rows
    rows = []
    sorted_p_indices = sorted(participant_data.keys())
    for p_idx in sorted_p_indices:
        data_lists = participant_data[p_idx]
        row = {'participant_index': p_idx}
        # Calculate Mean and STD for Energy
        energies = data_lists['min_energy']
        if energies:
            row['mean_min_energy'] = np.mean(energies)
            row['std_min_energy'] = np.std(energies)
            row['n_runs'] = len(energies)
        else:
            row['mean_min_energy'] = None
            row['std_min_energy'] = None
            row['n_runs'] = 0
        # Calculate Mean and STD for Params
        for p_key in param_keys:
            vals = data_lists[p_key]
            if vals:
                row[f'mean_{p_key}'] = np.mean(vals)
                row[f'std_{p_key}'] = np.std(vals)
            else:
                # Use standard NaN or None if parameter wasn't present
                row[f'mean_{p_key}'] = None
                row[f'std_{p_key}'] = None
        rows.append(row)
    # Create DataFrame
    df = pd.DataFrame(rows)
    # Reorder columns to put participant_index first, then energy stats, then params
    if not df.empty:
        cols = ['participant_index', 'n_runs', 'mean_min_energy', 'std_min_energy']
        # Add param cols
        for p_key in param_keys:
            cols.append(f'mean_{p_key}')
            cols.append(f'std_{p_key}')
        # Filter cols to only those that exist in df (e.g. if alpha was never found)
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    return df
target_file = '/Users/anchalbhaskar/Desktop/CBU Work/gnms/testingfile/gnm_index.json' # Defaulting to the test file
df = get_best_params_per_participant_multirun(target_file)
display(df.head())
output_csv = 'best_gnm_params_multirun.csv'
df.to_csv(output_csv, index=False)



