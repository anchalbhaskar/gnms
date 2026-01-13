#!/usr/bin/env python3

import argparse
from pathlib import Path
import os
import sys
import json

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.modules.pop("gnm", None)

from gnm import defaults, fitting, generative_rules, evaluation
from gnm.fitting.experiment_saving import ExperimentEvaluation


def parse_args():
    p = argparse.ArgumentParser(description="Run one GNM job (one eta/gamma + subject batch)")

    p.add_argument("--logdir", required=True)
    p.add_argument("--run-name", required=True)

    # iterating through subjects in batches
    p.add_argument("--subject-start", type=int, required=True)
    p.add_argument("--subject-count", type=int, required=True)

    # One (eta, gamma) per job
    p.add_argument("--eta", type=float, required=True)
    p.add_argument("--gamma", type=float, required=True)

    # Data paths
    p.add_argument("--connectivity-path", required=True)
    p.add_argument("--seed-path", required=True)

    # For local testing
    p.add_argument("--num-simulations", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()

    # Each job writes into its own folder
    outdir = Path(args.logdir) / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=== JOB START ===") #just for sanity checks and debugging --> JOB START
    print("outdir:", outdir)
    print("eta:", args.eta, "gamma:", args.gamma)
    print("subject batch:", args.subject_start, "count:", args.subject_count)
    print("device:", DEVICE)

    # ---- Distance matrix  ----
    distance_matrix = defaults.get_distance_matrix(device=DEVICE)

    # ---- Load connectomes ----
    weighted_connectivity = np.load(args.connectivity_path)
    # Expecting (90, 90, 446) -> transpose to (446, 90, 90)
    if weighted_connectivity.ndim != 3:
        raise ValueError(f"Expected 3D weighted_connectivity, got shape {weighted_connectivity.shape}")

    all_networks_np = np.transpose(weighted_connectivity, (2, 0, 1)).astype(np.float32)  # (S,90,90)
    all_binary_networks_np = (all_networks_np > 0).astype(np.float32)

    S, N, N2 = all_binary_networks_np.shape

    # ---- Slice subject batch   ----
    start = args.subject_start
    end = min(start + args.subject_count, S)

    if start < 0 or start >= S:
        raise ValueError(f"--subject-start {start} out of range (0..{S-1})")
    if end <= start:
        raise ValueError(f"Empty batch: start={start}, end={end}")

    batch_np = all_binary_networks_np[start:end]  # (B,90,90) #should be a subset of subjects
    real_binary_matrices = torch.tensor(batch_np, device=DEVICE)

    B = real_binary_matrices.shape[0]
    print(f"Loaded {S} subjects. Running batch [{start}:{end}) with B={B} subjects, N={N} nodes.")

    # ---- Compute num_connections PER SUBJECT  ----
    num_connections = int(real_binary_matrices.sum().item() / B)
    print(f"Avg connections per subject in this batch: {num_connections}")

    # ---- Seed ----
    seed = np.load(args.seed_path).astype(np.float32)  # (90,90)
    if seed.shape != (N, N):
        raise ValueError(f"Seed shape {seed.shape} does not match expected {(N, N)}")

    seed_new = np.broadcast_to(seed, (args.num_simulations, N, N))
    tensor_seed = torch.tensor(seed_new, device=DEVICE)

    # ---- one eta/gamma per job ----
    eta_values = torch.tensor([args.eta], device=DEVICE)
    gamma_values = torch.tensor([args.gamma], device=DEVICE)

    # ---- Sweep config  ----
    binary_sweep_parameters = fitting.BinarySweepParameters(
        eta=eta_values,
        gamma=gamma_values,
        lambdah=torch.tensor([0.0], device=DEVICE),
        distance_relationship_type=["powerlaw"],
        preferential_relationship_type=["powerlaw"],
        heterochronicity_relationship_type=["powerlaw"],
        generative_rule=[generative_rules.MatchingIndex()],
        num_iterations=[num_connections],
    )

    sweep_config = fitting.SweepConfig(
        binary_sweep_parameters=binary_sweep_parameters,
        num_simulations=args.num_simulations,
        distance_matrix=[distance_matrix],
        seed_weight_matrix=tensor_seed,
    )

    # ---- Evaluations --- 
    criteria = [
        evaluation.ClusteringKS(),
        evaluation.DegreeKS(),
        evaluation.EdgeLengthKS(distance_matrix),
        evaluation.BetweennessKS(),
    ]
    energy = evaluation.MaxCriteria(criteria)
    binary_evaluations = [energy]

    # ---- Run ----
    experiments = fitting.perform_sweep(
        sweep_config=sweep_config,
        binary_evaluations=binary_evaluations,
        real_binary_matrices=real_binary_matrices,
        save_model=False,
        save_run_history=False,
        verbose=True,
    )

    # ---- Save results INSIDE this job folder (important change) ----
    ev = ExperimentEvaluation(str(outdir))
    ev.save_experiments(experiments)

    # ---- Completion marker (so daemon can skip finished jobs --> like Francesco has in his file) ----
    (outdir / "model_last.pt").write_bytes(b"done\n")

    meta = {
        "eta": args.eta,
        "gamma": args.gamma,
        "subject_start": start,
        "subject_end": end,
        "batch_size": B,
        "num_connections_avg": num_connections,
        "num_simulations": args.num_simulations,
        "connectivity_path": args.connectivity_path,
        "seed_path": args.seed_path,
        "device": str(DEVICE),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("=== JOB DONE ===") #JOB DONE 


if __name__ == "__main__":
    main()