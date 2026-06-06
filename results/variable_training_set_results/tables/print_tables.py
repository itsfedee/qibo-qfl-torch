import json
import numpy as np
import os
import glob
import pandas as pd
from collections import defaultdict
from scipy.stats import median_abs_deviation


# =====================================================================
# Core: lettura dati
# =====================================================================

def _read_folder(folder, source="eval_metrics_client", save_json=False, seeds=None):
    """Legge tutti i JSON in una cartella, raggruppa per config, calcola mediana+MAD."""
    files = glob.glob(f"{folder}/*.json")
    if not files:
        print(f"  No files found in {folder}/")
        return {}

    if seeds is not None:
        seed_strs = {f"_seed{s}.json" for s in seeds}
        files = [f for f in files if any(f.endswith(s) for s in seed_strs)]
        if not files:
            print(f"  No files matching seeds={seeds} in {folder}/")
            return {}

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


def _read_centralized(folder, seeds=None, save_json=False):
    """Legge i JSON centralizzati e restituisce un data_dict compatibile."""
    files = glob.glob(f"{folder}/seed*.json")
    if not files:
        print(f"  No files found in {folder}/")
        return None

    if seeds is not None:
        seed_strs = {f"seed{s}.json" for s in seeds}
        files = [f for f in files if any(f.endswith(s) for s in seed_strs)]

    merged = defaultdict(lambda: {"loss": [], "accuracy": [], "f1": []})
    for filename in files:
        with open(filename) as f:
            data = json.load(f)
        for entry in data["epochs"]:
            e = entry["epoch"]
            if entry.get("eval_loss") is not None:
                merged[e]["loss"].append(entry["eval_loss"])
                merged[e]["accuracy"].append(entry.get("eval_accuracy", 0))
                merged[e]["f1"].append(entry.get("eval_f1", 0))

    epochs = sorted(merged.keys())
    result = {"rounds": epochs, "n_seeds": len(files)}
    for metric in ("loss", "accuracy", "f1"):
        arr = [np.array(merged[e][metric]) for e in epochs]
        result[f"{metric}_median"] = [float(np.median(a)) for a in arr]
        result[f"{metric}_mad"] = [float(median_abs_deviation(a)) for a in arr]

    if save_json:
        json_dir = os.path.join(folder, "aggregated")
        os.makedirs(json_dir, exist_ok=True)
        out = {}
        for i, e in enumerate(epochs):
            out[str(e)] = {}
            for m in ("loss", "accuracy", "f1"):
                out[str(e)][f"{m}_median"] = result[f"{m}_median"][i]
                out[str(e)][f"{m}_mad"] = result[f"{m}_mad"][i]
        out_path = os.path.join(json_dir, "centralized_aggregated.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=4)
        print(f"  Aggregated JSON saved to {out_path}")

    return result


# =====================================================================
# Tabella 1: confronto strategie
# =====================================================================

def print_summary(folder, configs, last_n=3, latex=True, filename="tabella_risultati.tex"):
    """Tabella riassuntiva delle strategie con loss, accuracy e F1."""
    import re
    agg_dir = os.path.join(folder, "aggregated")
    rows = []

    for strat, fname in configs.items():
        path = os.path.join(agg_dir, fname)
        if not os.path.exists(path):
            print(f"{strat}: FILE NOT FOUND: {fname}")
            continue

        with open(path) as f:
            data = json.load(f)

        rounds = sorted([int(r) for r in data.keys()])
        last = rounds[-last_n:]

        loss = np.mean([data[str(r)]["loss_median"] for r in last])
        loss_mad = np.mean([data[str(r)]["loss_mad"] for r in last])
        acc = np.mean([data[str(r)]["accuracy_median"] for r in last])
        acc_mad = np.mean([data[str(r)]["accuracy_mad"] for r in last])
        f1 = np.mean([data[str(r)]["f1_median"] for r in last])
        f1_mad = np.mean([data[str(r)]["f1_mad"] for r in last])

        config_str = fname.replace("_aggregated.json", "").split("_", 1)[1]

        # Step 1: separa i token (gli '_' qui sono solo separatori)
        config_latex = config_str.replace("_", ", ")
        # Step 2: sostituzioni LaTeX (etal PRIMA di eta!)
        config_latex = re.sub(r"etal([0-9.]+)", r"\\eta_l=\1", config_latex)
        config_latex = re.sub(r"eta([0-9.]+)", r"\\eta=\1", config_latex)
        config_latex = re.sub(r"mu([0-9.]+)", r"\\mu=\1", config_latex)
        config_latex = f"${config_latex}$"

        rows.append({
            "Strategy": strat,
            "Config": config_latex,
            "Loss": f"{loss:.3f} $\\pm$ {loss_mad:.3f}",
            "Accuracy (\\%)": f"{acc*100:.1f} $\\pm$ {acc_mad*100:.1f}",
            "F1 (\\%)": f"{f1*100:.1f} $\\pm$ {f1_mad*100:.1f}",
        })

    if not rows:
        print("  No data found.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if latex:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(df.to_latex(escape=False, index=False))
        print(f"\nLaTeX saved to {filename}")

    return df


# =====================================================================
# Tabella 2: noiseless / noisy / mitigated
# =====================================================================

def print_noise_summary(scenarios, source="eval_metrics_client", last_n=3,
                        mitigated=True, latex=True, filename="noise_table.tex", seeds=None):
    """Tabella noiseless/noisy(/mitigated) con recovery.

    scenarios: dict {label: cartella_o_data_dict}
    mitigated: se True, aggiunge la riga Recovery
    """
    rows = []
    loss_values = {}

    for label, folder in scenarios.items():
        if isinstance(folder, str):
            data = _read_folder(folder, source=source, seeds=seeds)
            if not data:
                continue
            data = list(data.values())[0]
        else:
            data = folder

        rounds = data["rounds"]
        last = rounds[-last_n:]
        idx = [rounds.index(r) for r in last]

        loss = np.mean([data["loss_median"][i] for i in idx])
        loss_mad = np.mean([data["loss_mad"][i] for i in idx])
        acc = np.mean([data["accuracy_median"][i] for i in idx])
        acc_mad = np.mean([data["accuracy_mad"][i] for i in idx])
        f1 = np.mean([data["f1_median"][i] for i in idx])
        f1_mad = np.mean([data["f1_mad"][i] for i in idx])

        loss_values[label] = loss
        rows.append({
            "Scenario": label,
            "Loss": f"{loss:.3f} $\\pm$ {loss_mad:.3f}",
            "Accuracy (\\%)": f"{acc*100:.1f} $\\pm$ {acc_mad*100:.1f}",
            "F1 (\\%)": f"{f1*100:.1f} $\\pm$ {f1_mad*100:.1f}",
        })

    if mitigated and len(loss_values) >= 3:
        labels = list(loss_values.keys())
        l_nl = loss_values[labels[0]]
        l_ny = loss_values[labels[1]]
        l_mt = loss_values[labels[2]]
        if l_ny != l_nl:
            recovery = (l_ny - l_mt) / (l_ny - l_nl) * 100
            rows.append({
                "Scenario": "\\textbf{Recovery}",
                "Loss": f"\\textbf{{{recovery:.0f}\\%}}",
                "Accuracy (\\%)": "---",
                "F1 (\\%)": "---",
            })

    if not rows:
        print("  No data found.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if latex:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(df.to_latex(escape=False, index=False))
        print(f"\nLaTeX saved to {filename}")

    return df


# =====================================================================
# Helper: statistiche centralizzate (prime max_epoch epoche, ultimi last_n)
# =====================================================================

def _centralized_stats(folder, max_epoch, last_n=3):
    """Ritorna dict con loss/acc/f1 median+mad sugli ultimi last_n delle prime max_epoch epoche."""
    centr = _read_centralized(folder)
    if centr is None:
        return None
    # filtra epoche <= max_epoch
    mask = [i for i, r in enumerate(centr["rounds"]) if r <= max_epoch]
    if len(mask) < last_n:
        return None
    idx = mask[-last_n:]
    return {
        "loss":     np.mean([centr["loss_median"][i] for i in idx]),
        "loss_mad": np.mean([centr["loss_mad"][i] for i in idx]),
        "acc":      np.mean([centr["accuracy_median"][i] for i in idx]),
        "acc_mad":  np.mean([centr["accuracy_mad"][i] for i in idx]),
        "f1":       np.mean([centr["f1_median"][i] for i in idx]),
        "f1_mad":   np.mean([centr["f1_mad"][i] for i in idx]),
    }


def _fmt(val, mad, pct=False):
    if pct:
        return f"{val*100:.1f} $\\pm$ {mad*100:.1f}"
    return f"{val:.3f} $\\pm$ {mad:.3f}"


def _format_config(fname):
    import re
    config_str = fname.replace("_aggregated.json", "").split("_", 1)[1]
    config_latex = config_str.replace("_", ", ")
    config_latex = re.sub(r"etal([0-9.]+)", r"\\eta_l=\1", config_latex)
    config_latex = re.sub(r"eta([0-9.]+)", r"\\eta=\1", config_latex)
    config_latex = re.sub(r"mu([0-9.]+)", r"\\mu=\1", config_latex)
    return f"${config_latex}$"


def _agg_stats(agg_dir, fname, last_n):
    """Legge un JSON aggregato e restituisce le statistiche sugli ultimi last_n round."""
    path = os.path.join(agg_dir, fname)
    if not os.path.exists(path):
        print(f"  FILE NOT FOUND: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    rounds = sorted([int(r) for r in data.keys()])
    last = rounds[-last_n:]
    return {
        "loss":     np.mean([data[str(r)]["loss_median"] for r in last]),
        "loss_mad": np.mean([data[str(r)]["loss_mad"] for r in last]),
        "acc":      np.mean([data[str(r)]["accuracy_median"] for r in last]),
        "acc_mad":  np.mean([data[str(r)]["accuracy_mad"] for r in last]),
        "f1":       np.mean([data[str(r)]["f1_median"] for r in last]),
        "f1_mad":   np.mean([data[str(r)]["f1_mad"] for r in last]),
    }


# =====================================================================
# Tabella combinata: confronto strategie IID + non-IID
# =====================================================================

def print_summary_combined(iid_folder, iid_configs, noniid_folder, noniid_configs,
                           centralized_folder=None, centralized_max_epoch=30,
                           last_n=3, latex=True, filename="tables/combined_simulations.tex"):
    """Tabella unica con colonne IID e non-IID, opzionalmente centralizzato in fondo."""
    iid_agg = os.path.join(iid_folder, "aggregated")
    noniid_agg = os.path.join(noniid_folder, "aggregated")

    # unione ordinata delle strategie
    all_strategies = list(iid_configs.keys())
    for s in noniid_configs:
        if s not in all_strategies:
            all_strategies.append(s)

    # raccogli dati: lista di (strategy, iid_stats, noniid_stats)
    data_rows = []
    for strat in all_strategies:
        iid_fname = iid_configs.get(strat)
        noniid_fname = noniid_configs.get(strat)
        iid_s = _agg_stats(iid_agg, iid_fname, last_n) if iid_fname else None
        noniid_s = _agg_stats(noniid_agg, noniid_fname, last_n) if noniid_fname else None
        data_rows.append((strat, iid_s, noniid_s))

    if centralized_folder:
        cs = _centralized_stats(centralized_folder, centralized_max_epoch, last_n)
        if cs:
            data_rows.append(("Centralized", cs, cs))

    if not data_rows:
        print("  No data found.")
        return

    # stampa a console
    def _row_str(strat, iid_s, noniid_s):
        parts = [f"{strat:15s}"]
        for s in (iid_s, noniid_s):
            if s:
                parts.append(f"{s['loss']:.3f}±{s['loss_mad']:.3f}  {s['acc']*100:.1f}±{s['acc_mad']*100:.1f}  {s['f1']*100:.1f}±{s['f1_mad']*100:.1f}")
            else:
                parts.append("---")
        return " | ".join(parts)

    print(f"{'Strategy':15s} | {'IID':>30s} | {'non-IID':>30s}")
    for strat, iid_s, noniid_s in data_rows:
        print(_row_str(strat, iid_s, noniid_s))

    # LaTeX
    if latex:
        def _cells(s):
            if s is None:
                return "--- & --- & ---"
            return f"{_fmt(s['loss'], s['loss_mad'])} & {_fmt(s['acc'], s['acc_mad'], pct=True)} & {_fmt(s['f1'], s['f1_mad'], pct=True)}"

        lines = []
        lines.append(r"\begin{tabular}{l ccc ccc}")
        lines.append(r"\toprule")
        lines.append(r" & \multicolumn{3}{c}{\textbf{IID}} & \multicolumn{3}{c}{\textbf{non-IID}} \\")
        lines.append(r"\cmidrule(r){2-4} \cmidrule(l){5-7}")
        lines.append(r"\textbf{Strategy} & \textbf{Loss} & \textbf{Acc (\%)} & \textbf{F1 (\%)} & \textbf{Loss} & \textbf{Acc (\%)} & \textbf{F1 (\%)} \\")
        lines.append(r"\midrule")
        for strat, iid_s, noniid_s in data_rows:
            lines.append(f"{strat} & {_cells(iid_s)} & {_cells(noniid_s)} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write("\n".join(lines))
        print(f"\nLaTeX saved to {filename}")


# =====================================================================
# Tabella combinata: noise/mitigati IID + non-IID
# =====================================================================

def print_noise_summary_combined(iid_scenarios, noniid_scenarios,
                                 source="eval_metrics_client", last_n=3,
                                 mitigated=True, centralized_folder=None,
                                 centralized_max_epoch=10,
                                 latex=True, filename="tables/combined_noise.tex",
                                 seeds=None):
    """Tabella noise combinata IID / non-IID, con recovery e centralizzato opzionale."""

    def _scenario_stats(scenarios, source, last_n, seeds):
        stats = {}
        for label, folder in scenarios.items():
            if isinstance(folder, str):
                data = _read_folder(folder, source=source, seeds=seeds)
                if not data:
                    continue
                data = list(data.values())[0]
            else:
                data = folder
            rounds = data["rounds"]
            last = rounds[-last_n:]
            idx = [rounds.index(r) for r in last]
            stats[label] = {
                "loss":     np.mean([data["loss_median"][i] for i in idx]),
                "loss_mad": np.mean([data["loss_mad"][i] for i in idx]),
                "acc":      np.mean([data["accuracy_median"][i] for i in idx]),
                "acc_mad":  np.mean([data["accuracy_mad"][i] for i in idx]),
                "f1":       np.mean([data["f1_median"][i] for i in idx]),
                "f1_mad":   np.mean([data["f1_mad"][i] for i in idx]),
            }
        return stats

    iid_stats = _scenario_stats(iid_scenarios, source, last_n, seeds)
    noniid_stats = _scenario_stats(noniid_scenarios, source, last_n, seeds)

    # unione ordinata degli scenari
    all_labels = list(iid_scenarios.keys())
    for l in noniid_scenarios:
        if l not in all_labels:
            all_labels.append(l)

    # raccogli dati: (label, iid_s, noniid_s)
    data_rows = []
    for label in all_labels:
        data_rows.append((label, iid_stats.get(label), noniid_stats.get(label)))

    # Recovery
    recovery_iid = None
    recovery_noniid = None
    if mitigated:
        for tag, stats in [("IID", iid_stats), ("non-IID", noniid_stats)]:
            labels = [l for l in all_labels if l in stats]
            if len(labels) >= 3:
                l_nl = stats[labels[0]]["loss"]
                l_ny = stats[labels[1]]["loss"]
                l_mt = stats[labels[2]]["loss"]
                if l_ny != l_nl:
                    recovery = (l_ny - l_mt) / (l_ny - l_nl) * 100
                    if tag == "IID":
                        recovery_iid = recovery
                    else:
                        recovery_noniid = recovery

    # Centralizzato
    cs = None
    if centralized_folder:
        cs = _centralized_stats(centralized_folder, centralized_max_epoch, last_n)

    if not data_rows:
        print("  No data found.")
        return

    # stampa a console
    print(f"{'Scenario':20s} | {'IID':>30s} | {'non-IID':>30s}")
    for label, iid_s, noniid_s in data_rows:
        parts = [f"{label:20s}"]
        for s in (iid_s, noniid_s):
            if s:
                parts.append(f"{s['loss']:.3f}±{s['loss_mad']:.3f}  {s['acc']*100:.1f}±{s['acc_mad']*100:.1f}  {s['f1']*100:.1f}±{s['f1_mad']*100:.1f}")
            else:
                parts.append("---")
        print(" | ".join(parts))
    if recovery_iid is not None or recovery_noniid is not None:
        print(f"{'Recovery':20s} | {f'{recovery_iid:.0f}%' if recovery_iid else '---':>30s} | {f'{recovery_noniid:.0f}%' if recovery_noniid else '---':>30s}")
    if cs:
        print(f"{'Centralized':20s} | {cs['loss']:.3f}±{cs['loss_mad']:.3f}  {cs['acc']*100:.1f}±{cs['acc_mad']*100:.1f}  {cs['f1']*100:.1f}±{cs['f1_mad']*100:.1f}")

    # LaTeX
    if latex:
        def _cells(s):
            if s is None:
                return "--- & --- & ---"
            return f"{_fmt(s['loss'], s['loss_mad'])} & {_fmt(s['acc'], s['acc_mad'], pct=True)} & {_fmt(s['f1'], s['f1_mad'], pct=True)}"

        lines = []
        lines.append(r"\begin{tabular}{l ccc ccc}")
        lines.append(r"\toprule")
        lines.append(r" & \multicolumn{3}{c}{\textbf{IID}} & \multicolumn{3}{c}{\textbf{non-IID}} \\")
        lines.append(r"\cmidrule(r){2-4} \cmidrule(l){5-7}")
        lines.append(r"\textbf{Scenario} & \textbf{Loss} & \textbf{Acc (\%)} & \textbf{F1 (\%)} & \textbf{Loss} & \textbf{Acc (\%)} & \textbf{F1 (\%)} \\")
        lines.append(r"\midrule")
        for label, iid_s, noniid_s in data_rows:
            lines.append(f"{label} & {_cells(iid_s)} & {_cells(noniid_s)} \\\\")
        if recovery_iid is not None or recovery_noniid is not None:
            r_iid = f"\\textbf{{{recovery_iid:.0f}\\%}}" if recovery_iid is not None else "---"
            r_noniid = f"\\textbf{{{recovery_noniid:.0f}\\%}}" if recovery_noniid is not None else "---"
            lines.append(f"\\textbf{{Recovery}} & {r_iid} & --- & --- & {r_noniid} & --- & --- \\\\")
        if cs:
            lines.append(f"Centralized & {_cells(cs)} & {_cells(cs)} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write("\n".join(lines))
        print(f"\nLaTeX saved to {filename}")


# =====================================================================
# Tabella combinata: noiseless vs noisy con degradazione
# =====================================================================

def print_degradation_combined(iid_noiseless, iid_noisy, noniid_noiseless, noniid_noisy,
                               source="eval_metrics_client", last_n=3,
                               latex=True, filename="tables/combined_degradation.tex",
                               seeds=None):
    """Tabella noiseless/noisy IID+non-IID con % di degradazione."""

    def _get_stats(folder):
        data = _read_folder(folder, source=source, seeds=seeds)
        if not data:
            return None
        data = list(data.values())[0]
        rounds = data["rounds"]
        idx = [rounds.index(r) for r in rounds[-last_n:]]
        return {
            "loss":     np.mean([data["loss_median"][i] for i in idx]),
            "loss_mad": np.mean([data["loss_mad"][i] for i in idx]),
            "acc":      np.mean([data["accuracy_median"][i] for i in idx]),
            "acc_mad":  np.mean([data["accuracy_mad"][i] for i in idx]),
            "f1":       np.mean([data["f1_median"][i] for i in idx]),
            "f1_mad":   np.mean([data["f1_mad"][i] for i in idx]),
        }

    iid_nl = _get_stats(iid_noiseless)
    iid_ny = _get_stats(iid_noisy)
    noniid_nl = _get_stats(noniid_noiseless)
    noniid_ny = _get_stats(noniid_noisy)

    def _degradation(nl, ny):
        if nl is None or ny is None:
            return None
        d = {}
        # loss: degradation = quanto è aumentata (positivo = peggio)
        if nl["loss"] != 0:
            d["loss"] = (ny["loss"] - nl["loss"]) / nl["loss"] * 100
        # acc/f1: degradation = quanto è diminuita (positivo = peggio)
        if nl["acc"] != 0:
            d["acc"] = (nl["acc"] - ny["acc"]) / nl["acc"] * 100
        if nl["f1"] != 0:
            d["f1"] = (nl["f1"] - ny["f1"]) / nl["f1"] * 100
        return d

    iid_deg = _degradation(iid_nl, iid_ny)
    noniid_deg = _degradation(noniid_nl, noniid_ny)

    data_rows = [
        ("Noiseless", iid_nl, noniid_nl),
        ("Noisy", iid_ny, noniid_ny),
    ]

    # console
    print(f"{'Scenario':15s} | {'IID':>35s} | {'non-IID':>35s}")
    for label, iid_s, noniid_s in data_rows:
        parts = [f"{label:15s}"]
        for s in (iid_s, noniid_s):
            if s:
                parts.append(f"{s['loss']:.3f}±{s['loss_mad']:.3f}  {s['acc']*100:.1f}±{s['acc_mad']*100:.1f}  {s['f1']*100:.1f}±{s['f1_mad']*100:.1f}")
            else:
                parts.append("---")
        print(" | ".join(parts))
    if iid_deg or noniid_deg:
        parts = [f"{'Degradation':15s}"]
        for d in (iid_deg, noniid_deg):
            if d:
                parts.append(f"L:+{d['loss']:.1f}%  A:-{d['acc']:.1f}%  F1:-{d['f1']:.1f}%")
            else:
                parts.append("---")
        print(" | ".join(parts))

    # LaTeX
    if latex:
        def _cells(s):
            if s is None:
                return "--- & --- & ---"
            return f"{_fmt(s['loss'], s['loss_mad'])} & {_fmt(s['acc'], s['acc_mad'], pct=True)} & {_fmt(s['f1'], s['f1_mad'], pct=True)}"

        def _deg_cells(d):
            if d is None:
                return "--- & --- & ---"
            return f"+{d['loss']:.1f}\\% & $-${d['acc']:.1f}\\% & $-${d['f1']:.1f}\\%"

        lines = []
        lines.append(r"\begin{tabular}{l ccc ccc}")
        lines.append(r"\toprule")
        lines.append(r" & \multicolumn{3}{c}{\textbf{IID}} & \multicolumn{3}{c}{\textbf{non-IID}} \\")
        lines.append(r"\cmidrule(r){2-4} \cmidrule(l){5-7}")
        lines.append(r"\textbf{Scenario} & \textbf{Loss} & \textbf{Acc (\%)} & \textbf{F1 (\%)} & \textbf{Loss} & \textbf{Acc (\%)} & \textbf{F1 (\%)} \\")
        lines.append(r"\midrule")
        for label, iid_s, noniid_s in data_rows:
            lines.append(f"{label} & {_cells(iid_s)} & {_cells(noniid_s)} \\\\")
        lines.append(f"\\textbf{{Degradation}} & {_deg_cells(iid_deg)} & {_deg_cells(noniid_deg)} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write("\n".join(lines))
        print(f"\nLaTeX saved to {filename}")


# =====================================================================

if __name__ == "__main__":


    centr = "centralized_model_results"
    print_noise_summary(
      scenarios={
          "Noiseless":   _read_centralized(f"{centr}/noiseless/nshots_1000"),
          "Noisy 0.005": _read_centralized(f"{centr}/noisy/p0.005/nshots_1000"),
          "Mitigated":   _read_centralized(f"{centr}/mitigated/p0.005/nshots_1000"),
      },
      source="eval_metrics_client",  # non usato perché passi già data_dict
      mitigated=True,
      filename="tables/centralized_noise_mit.tex",
  )
"""
    print_summary_combined(
      iid_folder="strategies_comparison/iid/noiseless/simulations/simulation_experiments",
      iid_configs={
          "FedAvg":     "fedavg_etal0.3_aggregated.json",
          "FedProx":    "fedprox_mu0.03_etal0.3_aggregated.json",
          "FedAdagrad": "fedadagrad_eta0.3_etal0.2_aggregated.json",
          "FedAdam":    "fedadam_eta0.2_etal0.15_aggregated.json",
          "FedYogi":    "fedyogi_eta0.1_etal0.1_aggregated.json",
      },
      noniid_folder="strategies_comparison/non_iid/noiseless/simulations/simulation_experiments",
      noniid_configs={
          "FedAvg":     "fedavg_etal0.3_aggregated.json",
          "FedProx":    "fedprox_mu0.03_etal0.3_aggregated.json",
          "FedAdagrad": "fedadagrad_eta0.3_etal0.2_aggregated.json",
          "FedAdam":    "fedadam_eta0.1_etal0.1_aggregated.json",
          "FedYogi":    "fedyogi_eta0.1_etal0.1_aggregated.json",
      },
      centralized_folder="centralized_model_results/noiseless/nshots_None",
      filename="tables/combined_comparison_simulations.tex",
  )


    print("---iid tuning table---")

    print_summary(
      folder="strategies_comparison/iid/noiseless/tuning/tuning_experiments",
      configs={
          "FedAvg":     "fedavg_etal0.3_aggregated.json",
          "FedProx":    "fedprox_mu0.03_etal0.3_aggregated.json",
          "FedAdagrad": "fedadagrad_eta0.3_etal0.2_aggregated.json",
          "FedAdam":    "fedadam_eta0.2_etal0.15_aggregated.json",
          "FedYogi":    "fedyogi_eta0.1_etal0.1_aggregated.json",
      },
      filename = "tables/iid/iid_tuning.tex",
  )
    

    
    print("---iid strategy comparison table---")
    
    print_summary(
      folder="strategies_comparison/iid/noiseless/simulations/simulation_experiments",
      configs={
          "FedAvg":     "fedavg_etal0.3_aggregated.json",
          "FedProx":    "fedprox_mu0.03_etal0.3_aggregated.json",
          "FedAdagrad": "fedadagrad_eta0.3_etal0.2_aggregated.json",
          "FedAdam":    "fedadam_eta0.2_etal0.15_aggregated.json",
          "FedYogi":    "fedyogi_eta0.1_etal0.1_aggregated.json",
      },
      filename = "tables/iid/iid_simulations.tex",
  )

    print ("---noisy values table ---")

    R = "results/iid/fedavg"

    print_noise_summary(
      scenarios={
          "Noiseless":   f"{R}/noiseless/uniform_p0.0_r0.0/nshots_1000",
          "Noisy 0.003": f"{R}/noisy/scaled_p0.003_r0.003_s0.002/nshots_1000",
          "Noisy 0.005": f"{R}/noisy/scaled_p0.005_r0.005_s0.002/nshots_1000",
          "Noisy 0.007": f"{R}/noisy/scaled_p0.007_r0.007_s0.002/nshots_1000",
      },
      source="eval_metrics_client",
      mitigated=False,
      filename="tables/iid/fedavg_noise_levels.tex",
  )
    
    print ("---noiseless-noisy-mit values table ---")

    print_noise_summary(
      scenarios={
          "Noiseless":   f"{R}/noiseless/uniform_p0.0_r0.0/nshots_1000",
          "Noisy 0.005": f"{R}/noisy/scaled_p0.005_r0.005_s0.002/nshots_1000",
          "Mitigated": f"{R}/mitigated/scaled_p0.005_r0.005_s0.002/nshots_1000",
      },
      source="eval_metrics_client",
      mitigated=True,
      filename="tables/iid/fedavg_noise_mit.tex",
  )
    
    print ("---mit values table ---")
    print_noise_summary(
      scenarios={
          "Noiseless":       f"{R}/noiseless/uniform_p0.0_r0.0/nshots_1000",
          "Mitigated 0.003": f"{R}/mitigated/scaled_p0.003_r0.003_s0.002/nshots_1000",
          "Mitigated 0.005": f"{R}/mitigated/scaled_p0.005_r0.005_s0.002/nshots_1000",
          "Mitigated 0.007": f"{R}/mitigated/scaled_p0.007_r0.007_s0.002/nshots_1000",
      },
      source="eval_metrics_client",
      mitigated=False,
      filename="tables/iid/fedavg_mit_levels.tex",
  )
    
    print ("---uniform noise table---")
    print_noise_summary(
      scenarios={
          "Noiseless":           f"{R}/noiseless/uniform_p0.0_r0.0/nshots_1000",
          "Noisy (uniform)":     f"{R}/noisy/uniform_p0.005_r0.005/nshots_1000",
          "Mitigated (uniform)": f"{R}/mitigated/uniform_p0.005_r0.005/nshots_1000",
      },
      source="eval_metrics_client",
      mitigated=True,
      filename="tables/iid/fedavg_uniform_levels.tex",
  )
    
    print ("---shot noise table---")
    print_noise_summary(
      scenarios={
          "Noisy 50 shots":   f"{R}/noisy/scaled_p0.005_r0.005_s0.002/nshots_50",
          "Noisy 500 shots":  f"{R}/noisy/scaled_p0.005_r0.005_s0.002/nshots_500",
          "Noisy 1000 shots": f"{R}/noisy/scaled_p0.005_r0.005_s0.002/nshots_1000",
          "Noisy 5000 shots": f"{R}/noisy/scaled_p0.005_r0.005_s0.002/nshots_5000",
      },
      source="eval_metrics_client",
      mitigated=False,
      filename="tables/iid/fedavg_shot_levels.tex",
  )
    
    
    print ("---other strategies---")

    for strat, folder in [("fedprox", "fedprox"), ("fedadagrad", "fedadagrad"),
                          ("fedadam", "fedadam"), ("fedyogi", "fedyogi")]:
        S = f"results/iid/{folder}"
        print_noise_summary(
            scenarios={
                "Noiseless":   f"{S}/noiseless/uniform_p0.0_r0.0/nshots_1000",
                "Noisy 0.005": f"{S}/noisy/scaled_p0.005_r0.005_s0.002/nshots_1000",
                "Mitigated":   f"{S}/mitigated/scaled_p0.005_r0.005_s0.002/nshots_1000",
            },
            source="eval_metrics_client",
            mitigated=True,
            filename=f"tables/iid/{strat}_noise_mit.tex",
        )

    print("---non iid tuning table---")

    print_summary(
      folder="strategies_comparison/non_iid/noiseless/tuning/tuning_experiments",
      configs={
          "FedAvg":     "fedavg_etal0.3_aggregated.json",
          "FedProx":    "fedprox_mu0.03_etal0.3_aggregated.json",
          "FedAdagrad": "fedadagrad_eta0.3_etal0.2_aggregated.json",
          "FedAdam":    "fedadam_eta0.1_etal0.1_aggregated.json",
          "FedYogi":    "fedyogi_eta0.1_etal0.1_aggregated.json",
      },
      filename = "tables/non_iid/non_iid_tuning.tex",
  )

    print("---non iid strategy comparison table---")

    print_summary(
      folder="strategies_comparison/non_iid/noiseless/simulations/simulation_experiments",
      configs={
          "FedAvg":     "fedavg_etal0.3_aggregated.json",
          "FedProx":    "fedprox_mu0.03_etal0.3_aggregated.json",
          "FedAdagrad": "fedadagrad_eta0.3_etal0.2_aggregated.json",
          "FedAdam":    "fedadam_eta0.1_etal0.1_aggregated.json",
          "FedYogi":    "fedyogi_eta0.1_etal0.1_aggregated.json",
      },
      filename = "tables/non_iid/non_iid_simulations.tex",
  )

    print("---non iid noiseless-noisy-mit FedAvg---")

    RN = "results/non_iid/fedavg"
    print_noise_summary(
      scenarios={
          "Noiseless":   f"{RN}/noiseless/uniform_p0.0_r0.0/nshots_1000",
          "Noisy 0.005": f"{RN}/noisy/scaled_p0.005_r0.005_s0.002/nshots_1000",
          "Mitigated":   f"{RN}/mitigated/scaled_p0.005_r0.005_s0.002/nshots_1000",
      },
      source="eval_metrics_client",
      mitigated=True,
      filename="tables/non_iid/fedavg_noise_mit.tex",
  )

    print("---non iid other strategies---")

    for strat, folder in [("fedprox", "fedprox"), ("fedadagrad", "fedadagrad"),
                          ("fedadam", "fedadam"), ("fedyogi", "fedyogi")]:
        S = f"results/non_iid/{folder}"
        print_noise_summary(
            scenarios={
                "Noiseless":   f"{S}/noiseless/uniform_p0.0_r0.0/nshots_1000",
                "Noisy 0.005": f"{S}/noisy/scaled_p0.005_r0.005_s0.002/nshots_1000",
                "Mitigated":   f"{S}/mitigated/scaled_p0.005_r0.005_s0.002/nshots_1000",
            },
            source="eval_metrics_client",
            mitigated=True,
            filename=f"tables/non_iid/{strat}_noise_mit.tex",
        )

    """