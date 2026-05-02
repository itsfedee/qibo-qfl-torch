import subprocess
import os
import time
from qibo_qfl_pt.experiment_utils import ExperimentPath

# =====================================================================
# Strategie e iperparametri
# =====================================================================
# Formato: strategy -> (server_param, server_val, client_param, client_val, seeds)

iid_strategies = {
    "FedAvg":     (None,  None,   "eta_l", 0.3,    [1,2,3,4,5,6,7]),
    "FedProx":    ("mu",  0.03,   "eta_l", 0.3,    [1,2,3,4,5,6,7]),
    "FedAdagrad": ("eta", 0.3,    "eta_l", 0.2,    [1,2,3,4,5,6,7]),
    "FedAdam":    ("eta", 0.2,    "eta_l", 0.15,   [1,2,3,4,5,6,7]),
    "FedYogi":    ("eta", 0.1,    "eta_l", 0.1,    [1,2,3,4,5,6,7]),
}

non_iid_strategies = {
    "FedAvg":     (None,  None,     "eta_l", 0.35,    [1,2,3,4,5,6,7]),
    "FedProx":    ("mu",  0.15,    "eta_l", 0.34,   [1,2,3,4,5,6,7]),
    "FedAdagrad": ("eta", 0.39,    "eta_l", 0.09,   [1,2,3,4,5,6,7]),
    "FedAdam":    ("eta", 0.2,    "eta_l", 0.06,   [1,2,3,4,5,6,7]),
    "FedYogi":    ("eta", 0.2,     "eta_l", 0.25,    [1,2,3,4,5,6,7]),
}

# =====================================================================
# Esperimenti da lanciare
# =====================================================================
# Ogni entry è un dict con:
#   distribution, mode, base_pauli, base_readout, scale, nshots, strategies
#
# - scale=0 → noise uniforme (uguale per tutti i client)
# - scale>0 → noise variabile per client
#
# Commenta/decommenta le run che ti servono.

runs = [

        # --- Noiseless ---
    {
        "distribution": "iid",
        "mode": "noiseless",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.002,
        "nshots": 500,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },

    {
        "distribution": "iid",
        "mode": "noiseless",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.002,
        "nshots": 1000,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },

    # --- Noisy 0.005 ---

    # run con noise model tutti uguali (scale 0)

    {
        "distribution": "iid",
        "mode": "noisy",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.0,
        "nshots": 1000,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },

    # run all'aumentare degli shosts (500, 1000, 5000)

    {
        "distribution": "iid",
        "mode": "noisy",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.002,
        "nshots": 500,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },

    {
        "distribution": "iid",
        "mode": "noisy",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.002,
        "nshots": 1000,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },


    {
        "distribution": "iid",
        "mode": "noisy",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.002,
        "nshots": 5000,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },

    # --- Mitigated ---
    {
        "distribution": "iid",
        "mode": "mitigated",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.002,
        "nshots": 1000,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },

    # mitigato per noise uguali
    {
        "distribution": "iid",
        "mode": "mitigated",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.0,
        "nshots": 1000,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },
]

# =====================================================================
# SOLO RUN MANCANTI — cancellare dopo aver completato
# =====================================================================
runs = [
    # noisy / uniform / nshots_1000 — mancano seed 2, 6
    {
        "distribution": "iid",
        "mode": "noisy",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.0,
        "nshots": 1000,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },

    # noisy / scaled / nshots_500 — manca seed 6
    {
        "distribution": "iid",
        "mode": "noisy",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.002,
        "nshots": 500,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },

    # noisy / scaled / nshots_1000 — manca seed 6
    {
        "distribution": "iid",
        "mode": "noisy",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.002,
        "nshots": 1000,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },

    # mitigated / scaled / nshots_1000 — mancano seed 3,4,5,6,7
    {
        "distribution": "iid",
        "mode": "mitigated",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.002,
        "nshots": 1000,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },

    # mitigated / uniform / nshots_1000 — mancano tutti (1-7)
    {
        "distribution": "iid",
        "mode": "mitigated",
        "base_pauli": 0.005,
        "base_readout": 0.005,
        "scale": 0.0,
        "nshots": 1000,
        "strategies": {k: iid_strategies[k] for k in ["FedAvg"]},
    },
]

# Override seeds per run mancanti
runs[0]["strategies"]["FedAvg"] = (None, None, "eta_l", 0.3, [2, 6])
runs[1]["strategies"]["FedAvg"] = (None, None, "eta_l", 0.3, [6])
runs[2]["strategies"]["FedAvg"] = (None, None, "eta_l", 0.3, [6])
runs[3]["strategies"]["FedAvg"] = (None, None, "eta_l", 0.3, [3, 4, 5, 6, 7])
runs[4]["strategies"]["FedAvg"] = (None, None, "eta_l", 0.3, [1, 2, 3, 4, 5, 6, 7])


# =====================================================================
# Runner
# =====================================================================

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = ''
env['RAY_DEDUP_LOGS'] = '0'
env['RAY_PRINT_ACTOR_STACK_TRACES'] = '0'
env['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'


def kill_ray():
    subprocess.run(["ray", "stop", "--force"], capture_output=True)
    subprocess.run(["taskkill", "/F", "/IM", "ray.exe"], capture_output=True)
    subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq ray*"], capture_output=True)


# Conta totale run
total = sum(
    len(seeds)
    for run in runs
    for _, (_, _, _, _, seeds) in run["strategies"].items()
)
run_count = 0

for run in runs:
    noise_label = ExperimentPath.noise_label(
        run["base_pauli"], run["base_readout"], run["scale"]
    )

    for strategy, (srv_name, srv_val, cli_name, cli_val, seeds) in run["strategies"].items():
        save_path = ExperimentPath.build(
            distribution=run["distribution"],
            strategy=strategy,
            mode=run["mode"],
            noise=noise_label,
            nshots=run["nshots"],
        )

        print(f"\n{'='*60}")
        print(f"  {strategy} | {run['mode']} | {noise_label} | nshots={run['nshots']}")
        print(f"  → {save_path}")
        print(f"  Seeds: {seeds}")
        print(f"{'='*60}")

        for seed in seeds:
            run_count += 1

            parts = [
                f'strategy="{strategy}"',
                f'seed={seed}',
                f'mode="{run["mode"]}"',
                f'base-pauli={run["base_pauli"]}',
                f'base-readout={run["base_readout"]}',
                f'scale={run["scale"]}',
                f'nshots={run["nshots"]}',
                f'save-path="{save_path}"',
            ]
            if srv_name is not None:
                parts.append(f'{srv_name}={srv_val}')
            if cli_name is not None:
                parts.append(f'{cli_name}={cli_val}')
            if run["distribution"] == "non_iid":
                parts.append('iid=false')
                parts.append('alpha=1.8')

            config_string = " ".join(parts)
            print(f"\n  [{run_count}/{total}] seed={seed}")

            kill_ray()
            time.sleep(3)

            try:
                subprocess.run(
                    ["flwr", "run", ".", "--run-config", config_string, "--stream"],
                    env=env, timeout=7200,
                )
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT")
                kill_ray()
                time.sleep(10)

            kill_ray()
            time.sleep(5)

print(f"\n{'='*60}")
print(f"Completed {run_count} runs.")
print(f"{'='*60}")
