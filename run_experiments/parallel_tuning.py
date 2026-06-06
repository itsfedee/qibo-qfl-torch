import subprocess
import os
import sys
import time
import random
import argparse
import tempfile
import shutil
import platform
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================================================================
# Platform detection
# =====================================================================

IS_WINDOWS = platform.system() == "Windows"

# =====================================================================
# Argomenti da riga di comando
# =====================================================================
# Esempi d'uso:
#   python parallel_tuning.py --distribution non_iid --workers 8
#   python parallel_tuning.py --distribution iid --strategy FedProx --workers 4


# =====================================================================
# Griglia iperparametri per il tuning
# =====================================================================

SEEDS = [14, 71, 130]

# Griglia per strategie a 2 parametri (server_lr, client_lr)
grid_lr = [
    (0.001, 0.01),
    (0.01, 0.15), (0.03, 0.3),
    (0.1, 0.01), (0.2, 0.05),
    (0.1, 0.1), (0.2, 0.15), (0.3, 0.2),
]

configs = {
    "FedAvg":     [(None, "eta_l", v) for v in [0.001, 0.005, 0.01, 0.15, 0.2, 0.25, 0.3, 0.35]],
    "FedAdagrad": [("eta", s, "eta_l", c) for s, c in grid_lr],
    "FedAdam":    [("eta", s, "eta_l", c) for s, c in grid_lr],
    "FedYogi":    [("eta", s, "eta_l", c) for s, c in grid_lr],
    "FedProx":    [("mu",  s, "eta_l", c) for s, c in grid_lr],
}


# =====================================================================
# Helper cross-platform per kill processi e cleanup
# =====================================================================

def kill_proc_tree(pid):
    """Killa un processo e tutti i suoi discendenti."""
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass
    else:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except (ProcessLookupError, PermissionError):
            pass


def final_cleanup():
    """Pulizia finale: chiude ray e killa eventuali orfani."""
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True, timeout=30)
    except Exception:
        pass

    if IS_WINDOWS:
        for image in ("ray.exe", "raylet.exe", "gcs_server.exe"):
            subprocess.run(["taskkill", "/F", "/IM", image], capture_output=True)
    else:
        for pattern in ("raylet", "gcs_server", "ray::"):
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


# =====================================================================
# Funzione per eseguire un singolo job di tuning
# =====================================================================

def run_single_job(job, timeout_sec, stagger_max=0.0):
    if stagger_max > 0:
        time.sleep(random.uniform(0, stagger_max))

    config_parts = [
        f'strategy="{job["strategy"]}"',
        f'seed={job["seed"]}',
        f'data-seed=2',
        f'save-path="{job["save_path"]}"',
        f'save-weights=false',
        f'testing=false',
        f'model-type="{job["model_type"]}"',
    ]
    if job["distribution"] == "non_iid":
        config_parts.append('iid=false')
        config_parts.append('alpha=1.8')

    if job["srv_name"] is not None:
        config_parts.append(f'{job["srv_name"]}={job["srv_val"]}')
    config_parts.append(f'{job["cli_name"]}={job["cli_val"]}')

    config_string = " ".join(config_parts)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['RAY_DEDUP_LOGS'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'

    ray_tmp = tempfile.mkdtemp(prefix=f"ray_tuning_{job['strategy']}_seed{job['seed']}_")
    env['RAY_TMPDIR'] = ray_tmp

    job_label = f"{job['strategy']} {job['srv_name']}={job['srv_val']} {job['cli_name']}={job['cli_val']} seed={job['seed']}"
    start_time = time.time()
    print(f"  [START] {job_label}")

    popen_kwargs = dict(
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    if not IS_WINDOWS:
        popen_kwargs["start_new_session"] = True

    tag = f"[{job['strategy']} s{job['seed']}]"
    proc = None
    try:
        try:
            proc = subprocess.Popen(
                ["flwr", "run", ".", "--run-config", config_string, "--stream"],
                **popen_kwargs,
            )
            deadline = time.time() + timeout_sec
            for line in proc.stdout:
                line = line.rstrip()
                if any(k in line for k in ("[ROUND", "Server eval", "ERROR", "aggregate_", "FAIL")):
                    print(f"  {tag} {line}", flush=True)
                if time.time() > deadline:
                    raise subprocess.TimeoutExpired(proc.args, timeout_sec)
            proc.wait()
            elapsed = time.time() - start_time

            if proc.returncode == 0:
                print(f"  [DONE]  {job_label} ({elapsed:.0f}s)")
                return ("ok", job_label, elapsed)
            else:
                print(f"  [FAIL]  {job_label} ({elapsed:.0f}s) returncode={proc.returncode}")
                return ("fail", job_label, elapsed)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"  [TIMEOUT] {job_label} ({elapsed:.0f}s)")
            if proc and proc.pid:
                kill_proc_tree(proc.pid)
            return ("timeout", job_label, elapsed)

    finally:
        if proc and proc.poll() is None:
            kill_proc_tree(proc.pid)
        time.sleep(2)
        shutil.rmtree(ray_tmp, ignore_errors=True)


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution",
        type=str,
        nargs="+",
        default=["iid"],
        choices=["iid", "non_iid"],
        help="Distribuzione/i dati. Default: iid. Più valori possibili.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Se specificato, tuning solo per quella strategia.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Quante run lanciare in parallelo. Default: 1.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per singola run, in secondi. Default: 3600 (1 ora).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        nargs="+",
        default=["quantum"],
        choices=["quantum", "hybrid", "classical"],
        help="Tipo/i di modello. Default: quantum. Più valori possibili.",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Skippa i primi N job (per riprendere da dove si era interrotto).",
    )
    args = parser.parse_args()

    print(f">>> Platform: {platform.system()} ({platform.machine()})")
    print(f">>> Distributions: {args.distribution}")

    pool_stagger = 0.0 if args.workers == 1 else 5.0 * args.workers

    # Strategia
    if args.strategy is not None:
        if args.strategy not in configs:
            print(f"ERRORE: strategia '{args.strategy}' non riconosciuta.")
            print(f"Strategie valide: {list(configs.keys())}")
            sys.exit(1)
        strategies_to_use = [args.strategy]
        print(f">>> Strategia: {args.strategy}")
    else:
        strategies_to_use = list(configs.keys())
        print(f">>> Tutte le strategie")

    print(f">>> Model types: {args.model_type}")
    print(f">>> Workers: {args.workers}")
    print(f">>> Timeout: {args.timeout}s")
    print(f">>> Seeds: {SEEDS}\n")

    # =================================================================
    # Costruzione lista jobs
    # =================================================================
    
    jobs = []
    for dist in args.distribution:
        for mt in args.model_type:
            if mt == "quantum":
                save_path = f"fixed_training_set_results/strategy_comparison_fixed/quantum/3_layers/{dist}/tuning_3L/tuning_experiments"
            elif mt == "classical":
                save_path = f"fixed_training_set_results/strategy_comparison_fixed/classical/9_hidden/{dist}/tuning_9h/tuning_experiments"
            elif mt == "hybrid":
                save_path = f"fixed_training_set_results/strategy_comparison_fixed/hybrid/3_layers_6_hidden/{dist}/tuning_3L_6h/tuning_experiments"

            print(f">>> [{dist} {mt}] Save path: {save_path}")

            for strategy in strategies_to_use:
                for config in configs[strategy]:
                    if config[0] is None:
                        srv_name, srv_val = None, None
                        cli_name, cli_val = config[1], config[2]
                    else:
                        srv_name, srv_val = config[0], config[1]
                        cli_name, cli_val = config[2], config[3]

                    for seed in SEEDS:
                        jobs.append({
                            "strategy": strategy,
                            "seed": seed,
                            "srv_name": srv_name,
                            "srv_val": srv_val,
                            "cli_name": cli_name,
                            "cli_val": cli_val,
                            "save_path": save_path,
                            "distribution": dist,
                            "model_type": mt,
                        })

    total = len(jobs)

    # Resume: skippa i primi N job
    if args.resume_from > 0:
        jobs = jobs[args.resume_from:]
        print(f"Totale jobs: {total} (skippati i primi {args.resume_from}, rimasti {len(jobs)})\n")
    else:
        print(f"Totale jobs: {total}\n")

    remaining = len(jobs)

    # =================================================================
    # Esecuzione
    # =================================================================

    print(f"{'='*70}")
    print(f"Avvio {remaining} jobs ({args.workers} workers dopo warmup)")
    print(f"{'='*70}\n")

    start_global = time.time()
    results = []
    completed = args.resume_from

    if remaining == 0:
        sys.exit(0)

    try:
        # Warmup
        print(">>> WARMUP: primo job sequenziale")
        warmup_status, warmup_label, warmup_elapsed = run_single_job(
            jobs[0], args.timeout, stagger_max=0.0
        )
        results.append((warmup_status, warmup_label, warmup_elapsed))
        completed += 1
        print(f"  >>> Warmup done ({warmup_status}) in {warmup_elapsed:.0f}s")
        print(f"  >>> Progress: {completed}/{total}\n")

        # Pool
        pool_jobs = jobs[1:]

        if pool_jobs and args.workers > 1:
            print(f">>> POOL: {len(pool_jobs)} jobs su {args.workers} workers\n")
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_single_job, job, args.timeout, pool_stagger): job
                    for job in pool_jobs
                }
                for future in as_completed(futures):
                    completed += 1
                    try:
                        status, label, elapsed = future.result()
                        results.append((status, label, elapsed))
                        print(f"  >>> Progress: {completed}/{total}")
                    except Exception as e:
                        print(f"  [ERROR] {e}")
                        results.append(("error", str(e), 0))
        elif pool_jobs:
            print(f">>> SEQUENZIALE: {len(pool_jobs)} jobs\n")
            for job in pool_jobs:
                status, label, elapsed = run_single_job(job, args.timeout, stagger_max=0.0)
                results.append((status, label, elapsed))
                completed += 1
                print(f"  >>> Progress: {completed}/{total}")

        elapsed_global = time.time() - start_global

        # Sommario
        ok = sum(1 for r in results if r[0] == "ok")
        fail = sum(1 for r in results if r[0] == "fail")
        timeout_count = sum(1 for r in results if r[0] == "timeout")
        error = sum(1 for r in results if r[0] == "error")

        print(f"\n{'='*70}")
        print(f"COMPLETATO in {elapsed_global:.0f}s ({elapsed_global/3600:.2f} ore)")
        print(f"  OK:      {ok}/{total}")
        print(f"  FAIL:    {fail}/{total}")
        print(f"  TIMEOUT: {timeout_count}/{total}")
        print(f"  ERROR:   {error}/{total}")
        print(f"{'='*70}\n")

        if fail > 0 or timeout_count > 0 or error > 0:
            print("Jobs falliti/timeout:")
            for status, label, elapsed in results:
                if status in ("fail", "timeout", "error"):
                    print(f"  [{status.upper()}] {label}")

    finally:
        print("\n>>> Pulizia finale (Ray + processi orfani)...")
        final_cleanup()
        print(">>> Done.")
