import subprocess
import os
import time

from qibo_qfl_pt.logging_config import setup_logging
setup_logging()

env = os.environ.copy()
env["RAY_DEDUP_LOGS"] = "1"
env["CUDA_VISIBLE_DEVICES"] = ""
env["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


SAVE_PATH = "strategies_comparison/iid/noiseless/tuning/tuning_experiments"
SEEDS = [14, 71, 130]


def kill_ray():
    subprocess.run(["ray", "stop", "--force"], capture_output=True)
    subprocess.run(["taskkill", "/F", "/IM", "ray.exe"], capture_output=True)
    subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq ray*"], capture_output=True)


# Griglia per strategie a 2 parametri
grid_lr = [
    # basso/basso 
    (0.001, 0.01),
    # basso/alto 
    (0.01, 0.15), (0.03, 0.3),
    # alto/basso 
    (0.1, 0.01), (0.2, 0.05),
    # alto/alto
    (0.1, 0.1), (0.2, 0.15), (0.3, 0.2)
]



configs = {
   "FedAvg":     [(None, "eta_l", v) for v in [0.001, 0.005, 0.01, 0.15, 0.2, 0.25, 0.3, 0.35]],
   "FedAdagrad":    [("eta", s, "eta_l", c) for s, c in grid_lr],
   "FedAdam":    [("eta", s, "eta_l", c) for s, c in grid_lr],
   "FedYogi": [("eta", s, "eta_l", c) for s, c in grid_lr],
   "FedProx":    [("mu",  s, "eta_l", c) for s, c in grid_lr],
}

total_runs = sum(len(cfgs) * len(SEEDS) for cfgs in configs.values())
print(f"Manual tuning IID: {total_runs} runs total")
print(f"Save path: {SAVE_PATH}")

run_count = 0
for strategy, cfgs in configs.items():
    for config in cfgs:
        if config[0] is None:
            p_server, p_server_val = None, None
            p_client, c_val = config[1], config[2]
        else:
            p_server, p_server_val = config[0], config[1]
            p_client, c_val = config[2], config[3]

        for seed in SEEDS:
            run_count += 1
            config_parts = [
                f'strategy="{strategy}"',
                f'seed={seed}',
                f'save-path="{SAVE_PATH}"',
                f'save-weights=false',
            ]
            if p_server is not None:
                config_parts.append(f'{p_server}={p_server_val}')
            config_parts.append(f'{p_client}={c_val}')

            config_string = " ".join(config_parts)

            print(f"\n{'='*70}")
            print(f"  [{run_count}/{total_runs}] {strategy} | Seed {seed}")
            print(f"  Config: {config_string}")
            print(f"{'='*70}\n")

            kill_ray()
            time.sleep(3)

            try:
                result = subprocess.run(
                    ["flwr", "run", ".", "--run-config", config_string, "--stream"],
                    env=env, timeout=3600,
                )
                print(f"\n  DONE | exit: {result.returncode}")
            except subprocess.TimeoutExpired:
                print(f"\n  TIMEOUT (killing and continuing)")
                kill_ray()
                time.sleep(10)

            kill_ray()
            time.sleep(5)

print(f"\n{'='*70}")
print("All manual tuning runs completed!")
print(f"{'='*70}")


