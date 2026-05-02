"""Utility per costruire path strutturati e scoprire esperimenti salvati."""

from pathlib import Path
from collections import defaultdict
import json
import glob


class ExperimentPath:
    """Costruisce e parsa path strutturati per gli esperimenti.

    Schema: results/{distribution}/{strategy}/{mode}/{noise}/nshots_{N}/
    """

    ROOT = "results"

    @staticmethod
    def build(
        strategy: str,
        mode: str,
        noise: str,
        nshots: int | str,
        distribution: str = "iid",
        root: str | None = None,
    ) -> str:
        """Costruisce il save_path dai parametri dell'esperimento.

        Args:
            strategy: es. "fedavg", "fedadam", "fedyogi"
            mode: "noiseless", "noisy", "mitigated"
            noise: es. "uniform_p0.005_r0.005", "scaled_p0.005_r0.005_s0.002"
            nshots: es. 1000, 5000, "none"
            distribution: "iid" o "non_iid"
            root: directory radice (default: "results")

        Returns:
            Path come stringa, es. "results/iid/fedavg/noisy/uniform_p0.005_r0.005/nshots_1000"
        """
        root = root or ExperimentPath.ROOT
        parts = [
            root,
            distribution,
            strategy.lower(),
            mode,
            noise,
            f"nshots_{nshots}",
        ]
        return "/".join(parts)

    @staticmethod
    def parse(path: str) -> dict:
        """Estrae i parametri da un path strutturato.

        Returns:
            dict con chiavi: distribution, strategy, mode, noise, nshots
        """
        parts = Path(path).parts
        # Cerca il livello "nshots_*" e risale
        for i, part in enumerate(parts):
            if part.startswith("nshots_"):
                nshots_raw = part.replace("nshots_", "")
                try:
                    nshots_val = int(nshots_raw)
                except ValueError:
                    nshots_val = nshots_raw
                return {
                    "distribution": parts[i - 4],
                    "strategy": parts[i - 3],
                    "mode": parts[i - 2],
                    "noise": parts[i - 1],
                    "nshots": nshots_val,
                    "path": str(path),
                }
        return {"path": str(path)}

    @staticmethod
    def noise_label(base_pauli: float, base_readout: float, scale: float = 0) -> str:
        """Genera la stringa noise config dai parametri numerici.

        Args:
            base_pauli: errore Pauli base
            base_readout: errore readout base
            scale: variazione tra client (0 = uniforme)

        Returns:
            es. "uniform_p0.005_r0.005" o "scaled_p0.005_r0.005_s0.002"
        """
        if scale == 0:
            return f"uniform_p{base_pauli}_r{base_readout}"
        return f"scaled_p{base_pauli}_r{base_readout}_s{scale}"


def discover_experiments(root: str | None = None) -> list[dict]:
    """Scansiona la directory results e restituisce una lista di esperimenti trovati.

    Ogni elemento è un dict con: distribution, strategy, mode, noise, nshots, n_seeds, path

    Se pandas è disponibile, puoi convertire con:
        pd.DataFrame(discover_experiments())
    """
    root = root or ExperimentPath.ROOT
    experiments = []

    for nshots_dir in Path(root).rglob("nshots_*"):
        if not nshots_dir.is_dir():
            continue
        json_files = list(nshots_dir.glob("*.json"))
        if not json_files:
            continue

        info = ExperimentPath.parse(str(nshots_dir))
        if "distribution" not in info:
            continue

        # Conta i seed (file che matchano *_seed*.json)
        seed_files = [f for f in json_files if "_seed" in f.name]
        info["n_seeds"] = len(seed_files)
        info["n_files"] = len(json_files)
        experiments.append(info)

    # Ordina per avere output consistente
    experiments.sort(key=lambda x: x["path"])
    return experiments
