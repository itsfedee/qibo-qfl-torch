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
from qibo_qfl_pt.experiment_utils import ExperimentPath

# =====================================================================
# Platform detection (fatta una volta sola, top-level)
# =====================================================================

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"

# =====================================================================
# Argomenti da riga di comando
# =====================================================================
# Esempi d'uso:
#   python parallel_experiments.py --strategy FedProx --workers 4
#   python parallel_experiments.py --strategy FedAdam --workers 6


# =====================================================================
# Strategie e iperparametri
# Top-level: i worker spawn-ati su Windows li importano.
# =====================================================================

iid_strategies = {
    "FedAvg":     (None,  None,   "eta_l", 0.3,    [1, 2, 3, 4, 5, 6, 7]),
  #  "FedProx":    ("mu",  0.03,   "eta_l", 0.3,    [1, 2, 3, 4, 5, 6, 7]),
  #  "FedAdagrad": ("eta", 0.3,    "eta_l", 0.2,    [1, 2, 3, 4, 5, 6, 7]),
 # "FedAdam":    ("eta", 0.2,    "eta_l", 0.15,   [1, 2, 3, 4, 5, 6, 7]),
   # "FedYogi":    ("eta", 0.1,    "eta_l", 0.1,    [1, 2, 3, 4, 5, 6, 7]),
}

non_iid_strategies = {
    "FedAvg":     (None,  None,   "eta_l", 0.3,   [1, 2, 3, 4, 5, 6, 7]),
    "FedProx":    ("mu",  0.03,   "eta_l", 0.3,   [1, 2, 3, 4, 5, 6, 7]),
    "FedAdagrad": ("eta", 0.3,   "eta_l", 0.2,   [1, 2, 3, 4, 5, 6, 7]),
    "FedAdam":    ("eta", 0.1,    "eta_l", 0.1,   [1, 2, 3, 4, 5, 6, 7]),
    "FedYogi":    ("eta", 0.1,    "eta_l", 0.1,   [5]),
}


# =====================================================================
# Helper cross-platform per kill processi e cleanup
# =====================================================================

def kill_proc_tree(pid):
    """Killa un processo e tutti i suoi discendenti.
    Cross-platform. Su Windows usa taskkill /T, su Unix usa il process group."""
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass
    else:
        # Su Unix abbiamo creato un nuovo session group con start_new_session=True,
        # quindi tutti i discendenti condividono il process group ID = pid.
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1)
            # Se dopo SIGTERM e' ancora vivo, SIGKILL secco
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # gia' morto, ottimo
        except (ProcessLookupError, PermissionError):
            pass


def final_cleanup():
    """Pulizia finale: chiude ray e killa eventuali orfani.
    Cross-platform."""
    # Tentativo soft via comando ray
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True, timeout=30)
    except Exception:
        pass

    # Hard kill OS-specifico per orfani sopravvissuti
    if IS_WINDOWS:
        for image in ("ray.exe", "raylet.exe", "gcs_server.exe"):
            subprocess.run(["taskkill", "/F", "/IM", image], capture_output=True)
    else:
        # pkill ritorna != 0 se non trova nulla, ed e' OK: capture_output ingoia tutto
        for pattern in ("raylet", "gcs_server", "ray::"):
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


def raise_fd_limit_if_possible():
    """Su Unix con tanti workers serve alzare il limit dei file descriptor.
    Su Windows non serve (Windows non ha questo limite)."""
    if IS_WINDOWS:
        return
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65536, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f">>> File descriptor limit alzato: {soft} -> {target}")
    except Exception as e:
        print(f">>> WARN: impossibile alzare fd limit ({e}). Se hai workers>16, "
              f"fai 'ulimit -n 65536' prima di lanciare lo script.")


# =====================================================================
# Funzione per eseguire un singolo job
# Top-level: deve essere importabile dai worker (spawn su Windows).
# =====================================================================

def run_single_job(job, timeout_sec, stagger_max=0.0):
    # Stagger casuale per evitare che tutti i worker bombardino il SuperLink
    # nello stesso istante. Il SuperLink locale ci mette ~15s a essere
    # operativo a freddo, e il client flwr CLI ha un timeout di 15s sulla
    # connessione iniziale: con N client simultanei almeno uno sfora.
    if stagger_max > 0:
        time.sleep(random.uniform(0, stagger_max))

    seed = job["seed"]
    parts = [
        f'strategy="{job["strategy"]}"',
        f'seed={seed}',
        f'data-seed=2',
        f'init-seed={seed}',
        f'sampling-seed={seed}',
        f'mode="{job["mode"]}"',
        f'model-type="{job["model_type"]}"',
        f'base-pauli={job["base_pauli"]}',
        f'base-readout={job["base_readout"]}',
        f'scale={job["scale"]}',
        f'nshots="{job["nshots"]}"' if isinstance(job["nshots"], str) else (f'nshots={job["nshots"]}' if job["nshots"] is not None else 'nshots="none"'),
        f'save-path="{job["save_path"]}"',

    ]
    if job["srv_name"] is not None:
        parts.append(f'{job["srv_name"]}={job["srv_val"]}')
    if job["cli_name"] is not None:
        parts.append(f'{job["cli_name"]}={job["cli_val"]}')
    if job["distribution"] == "non_iid":
        parts.append('iid=false')
        parts.append('alpha=1.8')

    config_string = " ".join(parts)

    # Ambiente custom per questo job.
    # NON impostiamo FLWR_HOME per-job: il SuperLink locale managed di
    # Flower e' uno solo per macchina, condiviso fra tutti i flwr run
    # paralleli, ed e' progettato per gestire run concorrenti.
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['RAY_DEDUP_LOGS'] = '0'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'

    # RAY_TMPDIR univoco per non incrociare i Ray work dirs fra job.
    ray_tmp = tempfile.mkdtemp(prefix=f"ray_{job['strategy']}_seed{seed}_")
    env['RAY_TMPDIR'] = ray_tmp

    job_label = (
        f"{job['strategy']} seed={seed} {job['mode']} "
        f"{job['noise_label']} nshots={job['nshots']}"
    )
    start_time = time.time()
    print(f"  [START] {job_label}")

    # Argomenti Popen cross-platform.
    # Su Unix start_new_session=True crea un nuovo process group, cosi'
    # possiamo killare tutti i discendenti via os.killpg.
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

    tag = f"[{job['strategy']} s{seed}]"
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
                # Stampa solo righe utili: round, metriche, errori
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
        # Sicurezza: se il processo per qualche motivo e' ancora vivo, killalo
        if proc and proc.poll() is None:
            kill_proc_tree(proc.pid)
        # Pausa per rilascio file handles (su Windows e' indispensabile,
        # su Unix non fa male)
        time.sleep(2)
        shutil.rmtree(ray_tmp, ignore_errors=True)


# =====================================================================
# Setup + esecuzione: TUTTO sotto __main__ (spawn su Windows).
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Se specificato, lancia solo quella strategia. Altrimenti tutte.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Quante run lanciare in parallelo. Default: 1."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout per singola run, in secondi. Default: 7200 (2 ore)."
    )
    args = parser.parse_args()

    print(f">>> Platform: {platform.system()} ({platform.machine()})")

    # Su Linux con tanti workers conviene alzare il fd limit
    if args.workers >= 8:
        raise_fd_limit_if_possible()

    # Stagger massimo all'avvio dei job nel pool. Per workers=1 non serve.
    # Per workers>=2 vogliamo distribuire l'avvio su qualche secondo cosi'
    # i client non bombardano il SuperLink simultaneamente.
    pool_stagger = 0.0 if args.workers == 1 else 5.0 * args.workers

    # Validazione strategia
    if args.strategy is not None:
        if args.strategy not in iid_strategies:
            print(f"ERRORE: strategia '{args.strategy}' non riconosciuta.")
            print(f"Strategie valide: {list(iid_strategies.keys())}")
            sys.exit(1)
        strategies_to_use = [args.strategy]
        print(f"\n>>> Modalità singola strategia: {args.strategy}")
    else:
        strategies_to_use = list(iid_strategies.keys())
        print(f"\n>>> Modalità completa: tutte le strategie")

    print(f">>> Workers paralleli: {args.workers}")
    print(f">>> Timeout per run: {args.timeout}s")
    print(f">>> Pool stagger max: {pool_stagger:.1f}s\n")

    # =================================================================
    # Esperimenti da lanciare: test delle altre strategie
    # =================================================================


    SEEDS = [1, 2, 3, 4, 5, 6, 7]

    # -----------------------------------------------------------------
    # Strategie per modello e distribuzione
    # -----------------------------------------------------------------

    # Quantum 3L
    quantum_iid = {
        "FedAvg":     (None,  None,  "eta_l", 0.3,   SEEDS),
        "FedAdagrad": ("eta", 0.3,   "eta_l", 0.2,   SEEDS),
        "FedAdam":    ("eta", 0.2,   "eta_l", 0.15,  SEEDS),
        "FedProx":    ("mu",  0.03,  "eta_l", 0.3,   SEEDS),
        "FedYogi":    ("eta", 0.1,   "eta_l", 0.1,   SEEDS),
    }
    quantum_non_iid = {
        "FedAvg":     (None,  None,  "eta_l", 0.3,   SEEDS),
        "FedAdagrad": ("eta", 0.3,   "eta_l", 0.2,   SEEDS),
        "FedAdam":    ("eta", 0.1,   "eta_l", 0.1,   SEEDS),
        "FedProx":    ("mu",  0.03,  "eta_l", 0.3,   SEEDS),
        "FedYogi":    ("eta", 0.1,   "eta_l", 0.1,   SEEDS),
    }

    # Hybrid 3L 6H
    hybrid_iid = {
        "FedAvg":     (None,  None,  "eta_l", 0.15,  SEEDS),
        "FedAdagrad": ("eta", 0.2,   "eta_l", 0.05,  SEEDS),
        "FedAdam":    ("eta", 0.1,   "eta_l", 0.1,   SEEDS),
        "FedProx":    ("mu",  0.1,   "eta_l", 0.1,   SEEDS),
        "FedYogi":    ("eta", 0.1,   "eta_l", 0.01,  SEEDS),
    }
    hybrid_non_iid = {
        "FedAvg":     (None,  None,  "eta_l", 0.25,  SEEDS),
        "FedAdagrad": ("eta", 0.2,   "eta_l", 0.15,  SEEDS),
        "FedAdam":    ("eta", 0.2,   "eta_l", 0.05,  SEEDS),
        "FedProx":    ("mu",  0.01,  "eta_l", 0.15,  SEEDS),
        "FedYogi":    ("eta", 0.1,   "eta_l", 0.1,   SEEDS),
    }

    # Classical 3H
    classical_iid_3h = {
        "FedAvg":     (None,  None,  "eta_l", 0.35,   SEEDS),
        "FedAdagrad": ("eta", 0.3,   "eta_l", 0.2,   SEEDS),
        "FedAdam":    ("eta", 0.2,   "eta_l", 0.15,  SEEDS),
        "FedProx":    ("mu",  0.03,  "eta_l", 0.3,   SEEDS),
        "FedYogi":    ("eta", 0.3,   "eta_l", 0.2,   SEEDS),
    }

    # Classical 9H
    classical_iid_9h = {
        "FedAvg":     (None,  None,  "eta_l", 0.35,  SEEDS),
        "FedAdagrad": ("eta", 0.3,   "eta_l", 0.2,   SEEDS),
        "FedAdam":    ("eta", 0.3,   "eta_l", 0.2,   SEEDS),
        "FedProx":    ("mu",  0.03,  "eta_l", 0.3,   SEEDS),
        "FedYogi":    ("eta", 0.2,   "eta_l", 0.15,  SEEDS),
    }
    classical_non_iid_9h = {
        "FedAvg":     (None,  None,  "eta_l", 0.35,  SEEDS),
        "FedAdagrad": ("eta", 0.3,   "eta_l", 0.2,   SEEDS),
        "FedAdam":    ("eta", 0.2,   "eta_l", 0.15,  SEEDS),
        "FedProx":    ("mu",  0.03,  "eta_l", 0.3,   SEEDS),
        "FedYogi":    ("eta", 0.2,   "eta_l", 0.01,  SEEDS),
    }

    # -----------------------------------------------------------------
    # Runs
    # -----------------------------------------------------------------

    runs = [
        # Quantum 3L - IID
        {"distribution": "iid", "mode": "noiseless",
         "model_type": "quantum",
         "base_pauli": 0.0, "base_readout": 0.0, "scale": 0.0,
         "nshots": "none",
         "save_path_override": "strategy_comparison_fixed_training/quantum/3_layers/iid/simulations/simulation_experiments",
         "strategies": quantum_iid},

        # Quantum 3L - Non-IID
        {"distribution": "non_iid", "mode": "noiseless",
         "model_type": "quantum",
         "base_pauli": 0.0, "base_readout": 0.0, "scale": 0.0,
         "nshots": "none",
         "save_path_override": "strategy_comparison_fixed_training/quantum/3_layers/non_iid/simulations/simulation_experiments",
         "strategies": quantum_non_iid},

        # Hybrid 3L 6H - IID
        {"distribution": "iid", "mode": "noiseless",
         "model_type": "hybrid",
         "base_pauli": 0.0, "base_readout": 0.0, "scale": 0.0,
         "nshots": "none",
         "save_path_override": "strategy_comparison_fixed_training/hybrid/3L_6hidden/iid/simulations/simulation_experiments",
         "strategies": hybrid_iid},

        # Hybrid 3L 6H - Non-IID
        {"distribution": "non_iid", "mode": "noiseless",
         "model_type": "hybrid",
         "base_pauli": 0.0, "base_readout": 0.0, "scale": 0.0,
         "nshots": "none",
         "save_path_override": "strategy_comparison_fixed_training/hybrid/3L_6hidden/non_iid/simulations/simulation_experiments",
         "strategies": hybrid_non_iid},


        # Classical 9H - IID
        {"distribution": "iid", "mode": "noiseless",
         "model_type": "classical",
         "base_pauli": 0.0, "base_readout": 0.0, "scale": 0.0,
         "nshots": "none",
         "save_path_override": "strategy_comparison_fixed_training/classical/9_hidden/iid/simulations/simulation_experiments",
         "strategies": classical_iid_9h},

        # Classical 9H - Non-IID
        {"distribution": "non_iid", "mode": "noiseless",
         "model_type": "classical",
         "base_pauli": 0.0, "base_readout": 0.0, "scale": 0.0,
         "nshots": "none",
         "save_path_override": "strategy_comparison_fixed_training/classical/9_hidden/non_iid/simulations/simulation_experiments",
         "strategies": classical_non_iid_9h},
    ]



    # =================================================================
    # Costruzione lista jobs (flatten)
    # =================================================================

    jobs = []
    for run in runs:
        noise_label = ExperimentPath.noise_label(
            run["base_pauli"], run["base_readout"], run["scale"]
        )
        for strategy, (srv_name, srv_val, cli_name, cli_val, seeds) in run["strategies"].items():
            save_path = run.get("save_path_override") or ExperimentPath.build(
                distribution=run["distribution"],
                strategy=strategy,
                mode=run["mode"],
                noise=noise_label,
                nshots=run["nshots"],
            )
            for seed in seeds:
                jobs.append({
                    "strategy": strategy,
                    "seed": seed,
                    "mode": run["mode"],
                    "model_type": run.get("model_type", "quantum"),
                    "base_pauli": run["base_pauli"],
                    "base_readout": run["base_readout"],
                    "scale": run["scale"],
                    "nshots": run["nshots"],
                    "save_path": save_path,
                    "srv_name": srv_name,
                    "srv_val": srv_val,
                    "cli_name": cli_name,
                    "cli_val": cli_val,
                    "distribution": run["distribution"],
                    "noise_label": noise_label,
                })

    total = len(jobs)
    print(f"Totale jobs da eseguire: {total}\n")

    # =================================================================
    # Esecuzione: warmup sequenziale + pool
    # =================================================================

    print(f"{'='*70}")
    print(f"Avvio {total} jobs ({args.workers} workers paralleli dopo warmup)")
    print(f"{'='*70}\n")

    start_global = time.time()
    results = []
    completed = 0

    if total == 0:
        sys.exit(0)

    try:
        # ---- Warmup sequenziale ----
        # Il primo job parte da solo, in modo che il SuperLink locale di Flower
        # abbia il tempo di startarsi (~15s a freddo) senza che altri client
        # ci finiscano sopra in race condition. Tutti i job successivi nel pool
        # si connetteranno allo stesso SuperLink ormai caldo.
        print(">>> WARMUP: avvio del primo job in sequenza per startare il SuperLink")
        warmup_status, warmup_label, warmup_elapsed = run_single_job(
            jobs[0], args.timeout, stagger_max=0.0
        )
        results.append((warmup_status, warmup_label, warmup_elapsed))
        completed += 1
        print(f"  >>> Warmup done ({warmup_status}) in {warmup_elapsed:.0f}s")
        print(f"  >>> Progress: {completed}/{total} done\n")

        # ---- Pool sui restanti ----
        pool_jobs = jobs[1:]

        if pool_jobs and args.workers > 1:
            print(f">>> POOL: {len(pool_jobs)} jobs restanti su {args.workers} workers\n")
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
                        print(f"  >>> Progress: {completed}/{total} done")
                    except Exception as e:
                        print(f"  [ERROR] {e}")
                        results.append(("error", str(e), 0))
        elif pool_jobs:
            # workers=1: tutto in sequenza, niente pool
            print(f">>> SEQUENZIALE: {len(pool_jobs)} jobs restanti su 1 worker\n")
            for job in pool_jobs:
                status, label, elapsed = run_single_job(job, args.timeout, stagger_max=0.0)
                results.append((status, label, elapsed))
                completed += 1
                print(f"  >>> Progress: {completed}/{total} done")

        elapsed_global = time.time() - start_global

        # =================================================================
        # Sommario finale
        # =================================================================

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
        # Pulizia finale, ESEGUITA SEMPRE — anche se l'utente fa Ctrl+C,
        # anche se qualcosa esplode dentro il pool.
        print("\n>>> Pulizia finale (Ray + processi orfani)...")
        final_cleanup()
        print(">>> Done.")
