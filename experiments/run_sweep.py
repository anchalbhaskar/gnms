import numpy as np

# =========================
# sweep configuration
# =========================

BATCH_SIZE = 50          # subjects per job
N_SUBJECTS = 446         # total subjects in your dataset

eta_list = [float(x) for x in np.linspace(-5, 0, 25)]
gamma_list = [float(x) for x in np.linspace(0, 1, 25)]
subject_starts = list(range(0, N_SUBJECTS, BATCH_SIZE))

CONFIGS = {
    "eta_gamma_batched": {
        "base_args": {
            # arguments are passed to every single run.py call
            "connectivity_path": "./weighted_connectivity.npy",
            "seed_path": "./seed.npy",
            "num_simulations": 10,
            "subject_count": BATCH_SIZE,
        },
        "grid": [
            {
                # Every combination of these is a separate job
                "eta": eta_list,
                "gamma": gamma_list,
                "subject_start": subject_starts,
            }
        ],
    }
}
