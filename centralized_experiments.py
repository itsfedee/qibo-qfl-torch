"""
Esperimenti centralizzati (singolo modello, no federazione).
Sweep su noise levels, salva JSON con metriche e pesi.

Uso:
  python centralized_experiments.py --workers 8
  python centralized_experiments.py --mode noisy --workers 4
  python centralized_experiments.py --mode mitigated --workers 4
"""

import os
import json
import time
import gc
import argparse
import platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# Importa da qibo_qfl_pt (patch + funzioni condivise)
import qibo_qfl_pt.patches  # noqa: F401 — applica i patch PSR e CDR
from qibo_qfl_pt.task import (
    set_seed, create_model, get_weights, NQUBITS,
)

import qibo
from qibo import set_backend, construct_backend
from qibo.noise import NoiseModel, PauliError, ReadoutError, gates


# =====================================================================
# Funzioni specifiche per il centralizzato
# =====================================================================

def generate_data(ndata=100, seed=42):
    """Genera dati circle-in-square come tensori PyTorch."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(ndata, 2))
    y = (np.linalg.norm(x, axis=1) <= 1.0).astype(np.float64)
    return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)


def build_noise_model_centralized(pauli_prob, readout_prob):
    """Noise model senza partition_id/scale (centralizzato)."""
    noise = NoiseModel()
    for q in range(NQUBITS):
        noise.add(
            PauliError([("X", pauli_prob), ("Y", pauli_prob), ("Z", pauli_prob)]),
            qubits=q,
        )
    single = np.array([
        [1 - readout_prob, readout_prob],
        [readout_prob, 1 - readout_prob],
    ])
    readout_matrix = np.kron(single, single)
    noise.add(ReadoutError(readout_matrix), gate=gates.M, qubits=[0, 1])
    return noise


def build_mitigation_config(readout_prob):
    """Config CDR per mitigazione."""
    single = np.array([[1 - readout_prob, readout_prob],
                       [readout_prob, 1 - readout_prob]])
    response_matrix = np.kron(single, single)
    return {
        "threshold": 0.01,
        "min_iterations": 500,
        "method": "CDR",
        "method_kwargs": {
            "n_training_samples": 120,
            "nshots": 30000,
            "seed": 40,
            "readout": {"response_matrix": response_matrix},
        },
    }


def get_trainable_params(model):
    return np.concatenate([p.detach().cpu().numpy().flatten()
                           for p in model.parameters()])


def train_and_evaluate(model, ndata_train, seed, epochs, lr, batch_size,
                       ndata_eval=200, seed_eval=40, job_tag=""):
    """Traina e ritorna metriche per epoca + pesi finali."""
    x_train, y_train = generate_data(ndata=ndata_train, seed=seed)
    x_eval, y_eval = generate_data(ndata=ndata_eval, seed=seed_eval)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epochs_data = []
    t0 = time.time()

    # Epoca 0: valutazione modello non trainato
    model.eval()
    with torch.no_grad():
        y_prob = model(x_eval).squeeze(-1)
        eval_loss_0 = loss_fn(y_prob.float(), y_eval.float()).item()
    y_pred = (y_prob >= 0.5).float()
    acc_0 = float((y_pred == y_eval).float().mean().item())
    f1_0 = float(f1_score(y_eval.numpy(), y_pred.numpy(), average="macro"))
    epochs_data.append({
        "epoch": 0,
        "train_loss": None,
        "eval_loss": eval_loss_0,
        "eval_accuracy": acc_0,
        "eval_f1": f1_0,
    })

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(x_train))
        batch_losses = []
        for i in range(0, len(x_train), batch_size):
            b = idx[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[b]).squeeze(-1).float(), y_train[b].float())
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            y_prob = model(x_eval).squeeze(-1)
            print(f"  y_prob stats: min={y_prob.min():.4e}, max={y_prob.max():.4e}, "
                f"mean={y_prob.mean():.4f}, "
                f"saturated_low={(y_prob < 1e-5).float().mean().item():.2%}, "
                f"saturated_high={(y_prob > 1-1e-5).float().mean().item():.2%}")
            eval_loss = loss_fn(y_prob.float(), y_eval.float()).item()
        y_pred = (y_prob >= 0.5).float()
        acc = float((y_pred == y_eval).float().mean().item())
        f1 = float(f1_score(y_eval.numpy(), y_pred.numpy(), average="macro"))

        epochs_data.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_accuracy": acc,
            "eval_f1": f1,
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            tag = f"{job_tag}" if job_tag else ""
            print(f"    {tag} Epoch {epoch+1}/{epochs}  eval loss: {eval_loss:.4f}  eval acc: {acc:.4f}", flush=True)

    elapsed = time.time() - t0
    weights = get_trainable_params(model)
    return epochs_data, weights, elapsed


# =====================================================================
# Singolo job
# =====================================================================

def run_single_job(job):
    """Esegue un singolo esperimento centralizzato."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    qibo.config.log.setLevel("ERROR")
    set_backend("numpy")
    set_seed(job["seed"])

    if job["mode"] == "noiseless":
        print(f"  [START] noiseless seed={job['seed']}", flush=True)
    else:
        print(f"  [START] {job['mode']} p={job['pauli_prob']} seed={job['seed']}", flush=True)

    noise_model = None
    mitigation_config = None

    if job["mode"] in ("noisy", "mitigated"):
        noise_model = build_noise_model_centralized(job["pauli_prob"], job["readout_prob"])

    if job["mode"] == "mitigated":
        mitigation_config = build_mitigation_config(job["readout_prob"])

    nshots = job["nshots"]
    model = create_model(
        model_type=job.get("model_type", "quantum"),
        nshots=nshots,
        noise_model=noise_model,
        mitigation_config=mitigation_config,
    )

    if job["mode"] == "noiseless":
        tag = f"[noiseless s{job['seed']}]"
    else:
        tag = f"[{job['mode']} p={job['pauli_prob']} s{job['seed']}]"

    epochs_data, weights, elapsed = train_and_evaluate(
        model,
        ndata_train=job["ndata"],
        seed=job["seed"],
        epochs=job["epochs"],
        lr=job["lr"],
        batch_size=job["batch_size"],
        job_tag=tag,
    )

    # Salva JSON
    result = {
        "info": {
            "mode": job["mode"],
            "seed": job["seed"],
            "pauli_prob": job["pauli_prob"],
            "readout_prob": job["readout_prob"],
            "nshots": job["nshots"],
            "epochs": job["epochs"],
            "lr": job["lr"],
            "batch_size": job["batch_size"],
            "ndata": job["ndata"],
            "elapsed_seconds": elapsed,
        },
        "epochs": epochs_data,
    }

    save_dir = Path(job["save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"lr_{job['lr']}_seed{job['seed']}"
    json_path = save_dir / f"{file_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=3)

    # Salva pesi
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.savez(weights_dir / f"{file_prefix}.npz", weights=weights)

    label = f"{job['mode']} lr={job['lr']} p={job['pauli_prob']} seed={job['seed']}"
    print(f"  [DONE]  {label} ({elapsed:.0f}s)")

    del model
    gc.collect()
    return ("ok", label, elapsed)


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None,
                        choices=["noiseless", "noisy", "mitigated"],
                        help="Se specificato, solo quel modo. Altrimenti tutti.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, nargs="+", default=[0.3])
    parser.add_argument("--ndata", type=int, default=500)
    parser.add_argument("--nshots", type=str, default="1000")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4])
    parser.add_argument("--model-type", type=str, default="quantum",
                        choices=["quantum", "hybrid", "classical"],
                        help="Tipo di modello: quantum, hybrid, classical.")
    parser.add_argument("--save-root", type=str, default="centralized_results_TEST")
    args = parser.parse_args()
    args.nshots = None if args.nshots.lower() == "none" else int(args.nshots)

    print(f">>> Platform: {platform.system()}")
    print(f">>> Workers: {args.workers}")
    print(f">>> Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f">>> LR values: {args.lr}")
    print(f">>> Seeds: {args.seeds}")
    print(f">>> Nshots: {args.nshots}")
    print(f">>> Model type: {args.model_type}\n")

    # Noise levels per lo sweep
    noise_levels = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.06]
    noise_levels = [0.005]

    modes = [args.mode] if args.mode else ["noiseless", "noisy", "mitigated"]

    # Costruisci jobs
    jobs = []
    for mode in modes:
        if mode == "noiseless":
            p_list = [0.0]
        else:
            p_list = [p for p in noise_levels if p > 0]

        for p in p_list:
            for lr in args.lr:
                if p == 0.0:
                    save_path = f"{args.save_root}/noiseless/nshots_{args.nshots}"
                else:
                    save_path = f"{args.save_root}/{mode}/p{p}/nshots_{args.nshots}"

                for seed in args.seeds:
                    # nome file: lr_{lr}_seed{seed}.json
                    jobs.append({
                        "mode": mode,
                        "seed": seed,
                        "pauli_prob": p,
                        "readout_prob": p,
                        "nshots": args.nshots,
                        "epochs": args.epochs,
                        "lr": lr,
                        "batch_size": args.batch_size,
                        "ndata": args.ndata,
                        "save_path": save_path,
                        "model_type": args.model_type,
                    })

    total = len(jobs)
    print(f"Totale jobs: {total}\n")

    start_global = time.time()
    results = []

    if args.workers > 1:
        print(f">>> POOL: {total} jobs su {args.workers} workers\n")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single_job, job): job for job in jobs}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    status, label, elapsed = future.result()
                    results.append((status, label, elapsed))
                    print(f"  >>> Progress: {i}/{total}")
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    results.append(("error", str(e), 0))
    else:
        print(f">>> SEQUENZIALE: {total} jobs\n")
        for i, job in enumerate(jobs, 1):
            try:
                status, label, elapsed = run_single_job(job)
                results.append((status, label, elapsed))
                print(f"  >>> Progress: {i}/{total}")
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("error", str(e), 0))

    elapsed_global = time.time() - start_global
    ok = sum(1 for r in results if r[0] == "ok")
    skip = sum(1 for r in results if r[0] == "skip")
    fail = sum(1 for r in results if r[0] not in ("ok", "skip"))

    print(f"\n{'='*70}")
    print(f"COMPLETATO in {elapsed_global:.0f}s ({elapsed_global/3600:.2f} ore)")
    print(f"  OK:    {ok}/{total}")
    print(f"  SKIP:  {skip}/{total}")
    print(f"  FAIL:  {fail}/{total}")
    print(f"{'='*70}")
