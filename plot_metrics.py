import json
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from collections import defaultdict
from scipy.stats import median_abs_deviation
import glob

import torch
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

from qibo_qfl_pt.task import create_model, set_weights, build_noise_model

warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================================
# Core
# =====================================================================

METRIC_INFO = {
    "loss":     {"ylabel": "Loss",         "pct": False, "title": "Loss"},
    "accuracy": {"ylabel": "Accuracy (%)", "pct": True,  "title": "Accuracy"},
    "f1":       {"ylabel": "F1",           "pct": True,  "title": "F1 score"},
}

COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan',
          'darkblue', 'darkred', 'darkgreen']


def _read_folder(folder, source="eval_metrics_client", save_json=False):
    """Legge tutti i JSON in una cartella, raggruppa per config, calcola mediana+MAD."""
    files = glob.glob(f"{folder}/*.json")
    if not files:
        print(f"  No files found in {folder}/")
        return {}

    # raggruppa per config (togliendo _seedN.json)
    groups = defaultdict(list)
    for f in files:
        name = os.path.basename(f)
        parts = name.rsplit("_seed", 1)
        key = parts[0] if len(parts) == 2 else name.replace(".json", "")
        groups[key].append(f)

    results = {}
    for key, file_list in groups.items():
        merged = defaultdict(lambda: {"loss": [], "accuracy": [], "f1": []})
        for filename in file_list:
            with open(filename) as f:
                data = json.load(f)
            for entry in data["rounds"]:
                r = entry["round"]
                m = entry.get(source, {})
                if not m:
                    continue
                merged[r]["loss"].append(m["loss"])
                merged[r]["accuracy"].append(m["accuracy"])
                merged[r]["f1"].append(m.get("f1_score", m.get("f1", 0)))

        rounds = sorted(merged.keys())
        result = {"rounds": rounds, "n_seeds": len(file_list)}
        for metric in ("loss", "accuracy", "f1"):
            arr_per_round = [np.array(merged[r][metric]) for r in rounds]
            result[f"{metric}_median"] = [float(np.median(a)) for a in arr_per_round]
            result[f"{metric}_mad"] = [float(median_abs_deviation(a)) for a in arr_per_round]
        results[key] = result
        print(f"  {key}: {len(file_list)} seeds")

        if save_json:
            json_dir = os.path.join(folder, "aggregated")
            os.makedirs(json_dir, exist_ok=True)
            out = {}
            for i, r in enumerate(rounds):
                out[str(r)] = {}
                for m in ("loss", "accuracy", "f1"):
                    out[str(r)][f"{m}_median"] = result[f"{m}_median"][i]
                    out[str(r)][f"{m}_mad"] = result[f"{m}_mad"][i]
            with open(os.path.join(json_dir, f"{key}_aggregated.json"), "w") as f:
                json.dump(out, f, indent=4)

    return results


# =====================================================================
# Metriche
# =====================================================================

def plot(scenarios, save_path, source="eval_metrics_client", metrics=("loss",), prefix="", save_json=False):
    """
    Plotta confronto tra scenari.

    scenarios: dict {label: cartella} oppure dict {label: data_dict} (output di _read_folder)
    """
    all_data = {}
    for label, val in scenarios.items():
        if isinstance(val, str):
            folder_data = _read_folder(val, source=source, save_json=save_json)
            if len(folder_data) == 1:
                all_data[label] = list(folder_data.values())[0]
            else:
                for key, data in folder_data.items():
                    all_data[f"{label} - {key}" if label and len(folder_data) > 1 else key] = data
        else:
            all_data[label] = val

    if not all_data:
        print("No data to plot.")
        return

    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for idx, (label, data) in enumerate(all_data.items()):
        color = COLORS[idx % len(COLORS)]
        for ax_idx, metric in enumerate(metrics):
            info = METRIC_INFO[metric]
            vals = list(data[f"{metric}_median"])
            mads = list(data[f"{metric}_mad"])
            if info["pct"]:
                vals = [v * 100 for v in vals]
                mads = [m * 100 for m in mads]

            axes[ax_idx].plot(data["rounds"], vals, color=color, label=label, linewidth=2)
            if any(m > 0 for m in mads):
                axes[ax_idx].fill_between(
                    data["rounds"],
                    [v - m for v, m in zip(vals, mads)],
                    [v + m for v, m in zip(vals, mads)],
                    color=color, alpha=0.3,
                )

    for ax_idx, metric in enumerate(metrics):
        info = METRIC_INFO[metric]
        axes[ax_idx].set_xlabel("Round", fontsize=14)
        axes[ax_idx].set_ylabel(info["ylabel"], fontsize=14)
        axes[ax_idx].set_title(f"{prefix} {info['title']}".strip(), fontsize=16, fontweight="bold")
        axes[ax_idx].legend(fontsize=10)
        axes[ax_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.show()


def plot_tuning(folder, save_dir=None, source="eval_metrics_server", metrics=("loss",), save_json=False):
    """
    Plotta il tuning: un grafico per strategia, una curva per config di iperparametri.

    folder: cartella con tutti i JSON di tuning
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(folder), "plots")

    all_data = _read_folder(folder, source=source, save_json=save_json)
    if not all_data:
        return

    # raggruppa per strategia
    by_strategy = defaultdict(dict)
    for key, data in all_data.items():
        strategy = key.split("_")[0]
        params = key[len(strategy) + 1:]
        by_strategy[strategy][params] = data

    for strategy, configs in by_strategy.items():
        print(f"\n--- {strategy} ---")
        plot(
            configs,
            save_path=os.path.join(save_dir, f"{strategy}_tuning.png"),
            source=source,
            metrics=metrics,
            prefix=f"{strategy.capitalize()} Tuning",
        )


# =====================================================================
# Partizioni
# =====================================================================

def _generate_data(n_data, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n_data, 2))
    y = (np.linalg.norm(x, axis=1) <= 1.0).astype(np.int32)
    return {"point": x.astype(np.float64), "label": y}


def plot_partitions(seed, alpha=1.0, n_data=500, num_clients=5, save_path=None):
    """Plotta la distribuzione dei dati per ogni client."""
    data = _generate_data(n_data, seed)
    dataset = Dataset.from_dict(data)
    partitioner = DirichletPartitioner(
        num_partitions=num_clients, alpha=alpha, partition_by="label", seed=seed
    )
    partitioner.dataset = dataset

    fig, axes = plt.subplots(1, num_clients + 1, figsize=(4 * (num_clients + 1), 4))

    # plot globale
    x_all = np.array(data["point"])
    y_all = np.array(data["label"])
    colors_all = ["red" if l == 1 else "blue" for l in y_all]
    axes[0].scatter(x_all[:, 0], x_all[:, 1], c=colors_all, s=10, alpha=0.6)
    axes[0].add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.5))
    axes[0].set_title(f"Global ({len(y_all)})\nc0={np.sum(y_all==0)}, c1={np.sum(y_all==1)}", fontsize=11)
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_aspect("equal")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for pid in range(num_clients):
        part = partitioner.load_partition(pid)
        part.set_format("numpy")
        x_p = np.array(part["point"])
        y_p = np.array(part["label"])
        n0 = int(np.sum(y_p == 0))
        n1 = int(np.sum(y_p == 1))
        colors = ["red" if l == 1 else "blue" for l in y_p]

        ax = axes[pid + 1]
        ax.scatter(x_p[:, 0], x_p[:, 1], c=colors, s=10, alpha=0.6)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.5))
        ax.set_title(f"Client {pid} ({len(y_p)})\nc0={n0}, c1={n1}", fontsize=11)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"Dirichlet partition (seed={seed}, alpha={alpha})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_partitions_grid(seeds, alpha=1.0, n_data=500, num_clients=5, save_path="partitions"):
    """Plotta la distribuzione per più seed in una griglia."""
    n_seeds = len(seeds)
    n_cols = num_clients + 1
    fig, axes = plt.subplots(n_seeds, n_cols, figsize=(3.5 * n_cols, 3.5 * n_seeds))
    if n_seeds == 1:
        axes = axes[np.newaxis, :]

    for row, seed in enumerate(seeds):
        data = _generate_data(n_data, seed)
        dataset = Dataset.from_dict(data)
        partitioner = DirichletPartitioner(
            num_partitions=num_clients, alpha=alpha, partition_by="label", seed=seed
        )
        partitioner.dataset = dataset

        x_all = np.array(data["point"])
        y_all = np.array(data["label"])
        colors_all = ["red" if l == 1 else "blue" for l in y_all]
        axes[row, 0].scatter(x_all[:, 0], x_all[:, 1], c=colors_all, s=8, alpha=0.5)
        axes[row, 0].add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1))
        axes[row, 0].set_title(f"Global\n{np.sum(y_all==0)}/{np.sum(y_all==1)}", fontsize=9)
        axes[row, 0].set_xlim(-1.1, 1.1)
        axes[row, 0].set_ylim(-1.1, 1.1)
        axes[row, 0].set_aspect("equal")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(f"seed={seed}", fontsize=11, fontweight="bold")

        for pid in range(num_clients):
            part = partitioner.load_partition(pid)
            part.set_format("numpy")
            x_p = np.array(part["point"])
            y_p = np.array(part["label"])
            n0 = int(np.sum(y_p == 0))
            n1 = int(np.sum(y_p == 1))
            colors = ["red" if l == 1 else "blue" for l in y_p]

            ax = axes[row, pid + 1]
            ax.scatter(x_p[:, 0], x_p[:, 1], c=colors, s=8, alpha=0.5)
            ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1))
            ax.set_title(f"C{pid} ({len(y_p)})\n{n0}/{n1}", fontsize=9)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"Dirichlet partitions (alpha={alpha})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


# =====================================================================
# Predizioni
# =====================================================================

def _predict(weights_path, noise_model=None, nshots=None):
    model = create_model(noise_model=noise_model, nshots=nshots)
    weights = list(np.load(weights_path).values())
    set_weights(model, weights)

    rng = np.random.default_rng(0)
    x_grid = torch.tensor(rng.uniform(-1, 1, size=(2500, 2)), dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        y_prob = model(x_grid).squeeze(-1).numpy()

    return x_grid.numpy(), y_prob


def _draw(ax, x_grid, y_prob, title, colorbar=True):
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.5))
    scatter = ax.scatter(x_grid[:, 0], x_grid[:, 1], c=y_prob, cmap="RdBu",
                         vmin=0, vmax=1, s=15, alpha=0.8)
    if colorbar:
        plt.colorbar(scatter, ax=ax, label="P(inside)", shrink=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_predictions(weights_path, title="Predictions", save_path=None,
                     noise_model=None, nshots=None):
    x_grid, y_prob = _predict(weights_path, noise_model=noise_model, nshots=nshots)
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw(ax, x_grid, y_prob, title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_predictions_grid(scenarios, save_path=None, ncols=None):
    """
    Plotta più predizioni in griglia.

    scenarios: lista di (weights_path, title) oppure (weights_path, title, noise_kwargs)
    noise_kwargs: dict con chiavi opzionali 'pauli_base', 'readout_base', 'nshots'
    """
    n = len(scenarios)
    if ncols is None:
        ncols = n if n <= 3 else 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, scenario in enumerate(scenarios):
        if len(scenario) == 3:
            weights_path, title, noise_kwargs = scenario
            nm, _ = build_noise_model(
                pauli_base=noise_kwargs.get("pauli_base", 0),
                readout_base=noise_kwargs.get("readout_base", 0),
                partition_id=0, scale=0,
            )
            nshots = noise_kwargs.get("nshots", 1000)
            x_grid, y_prob = _predict(weights_path, noise_model=nm, nshots=nshots)
        else:
            weights_path, title = scenario
            x_grid, y_prob = _predict(weights_path)

        _draw(axes[i], x_grid, y_prob, title)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


# =====================================================================

if __name__ == "__main__":
    plot(
        scenarios={"": "strategies_comparison/iid/noiseless/simulations/simulation_experiments"},
        save_path="strategies_comparison/iid/noiseless/simulations/simulation_plots/comparison.pdf",
        source="eval_metrics_server",
        metrics=("loss", "accuracy", "f1"),
        save_json=True,
    )
