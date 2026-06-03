"""Quantum Federated Learning Server with Qiboml (PyTorch) and Flower."""

import os
import gc
import numpy as np
from pathlib import Path

from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from qibo_qfl_pt.custom_strategy import fedavg, fedadam, fedyogi, fedadagrad, fedprox
from qibo_qfl_pt.task import create_model, evaluate_model, get_weights, set_weights, load_data_server, set_seed


app = ServerApp()

strategies = {
    "FedAvg": (fedavg, None),
    "FedAdam": (fedadam, "eta"),
    "FedAdagrad": (fedadagrad, "eta"),
    "FedYogi": (fedyogi, "eta"),
    "FedProx": (fedprox, "mu"),
}


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    num_rounds = context.run_config["num-server-rounds"]
    num_clients = context.run_config["num-clients"]
    iid_flag = context.run_config["iid"]
    strategy_name = context.run_config["strategy"]
    fraction_train = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    local_epochs = context.run_config["local-epochs"]
    seed = context.run_config["seed"]
    init_seed = int(context.run_config.get("init-seed", seed))
    data_seed = int(context.run_config.get("data-seed", seed))
    alpha = context.run_config["alpha"]

    mode = context.run_config.get("mode", "noiseless")

    eta = context.run_config.get("eta", 0.1)
    eta_l = context.run_config.get("eta_l", 0.1)
    mu = context.run_config.get("mu", 0.1)

    base_pauli = context.run_config.get("base-pauli", 0.0)
    base_readout = context.run_config.get("base-readout", 0.0)
    noise_scale = context.run_config.get("scale", 0.0)

    model_type = context.run_config.get("model-type", "quantum")

    set_seed(init_seed)

    # pesi iniziali
    model = create_model(model_type=model_type)
    arrays = ArrayRecord(get_weights(model))
    del model

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy_class, param_name = strategies[strategy_name]
    save_path = context.run_config.get("save-path", "federated_results/iid/default")

    nshots = context.run_config.get("nshots", 1000)
    if nshots == "none":
        nshots_info = None
    else:
        nshots_info = int(nshots)

    noise_info = None
    if mode != "noiseless":
        noise_info = {
            "base_pauli": base_pauli,
            "base_readout": base_readout,
            "noise_scale": noise_scale,
        }

    seed_label = context.run_config.get("seed-label", "") or None

    kwargs = {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "local_epochs": local_epochs,
        "seed": seed,
        "init_seed": init_seed,
        "data_seed": data_seed,
        "sampling_seed": int(context.run_config.get("sampling-seed", seed)),
        "run_id": context.run_config.get("run-id", None),
        "save_path": save_path,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "alpha": alpha,
        "iid": iid_flag,
        "eta_l": eta_l,
        "noise_info": noise_info,
        "nshots": nshots_info,
        "training_mode": mode,
        "model_type": model_type,
        "seed_label": seed_label,
    }

    if param_name is not None:
        if param_name == "eta":
            kwargs["eta"] = eta
            kwargs["eta_l"] = eta_l
        elif param_name == "mu":
            kwargs["mu"] = mu

    strategy = strategy_class(**kwargs)

    def global_evaluate(server_round: int, arrays: ArrayRecord, testing: bool) -> MetricRecord:
        """Evaluate on centralized validation or test set."""
        # con testing il seed è 40, con tuning è 41
        x_val, y_val = load_data_server(ndata=context.run_config["n-eval-data"], testing=testing)
        model = create_model(model_type=model_type)
        set_weights(model, arrays.to_numpy_ndarrays())
        loss, acc, f1 = evaluate_model(model, x_val, y_val)
        print(f"Server eval - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
        del model
        gc.collect()
        return MetricRecord({"accuracy": acc, "loss": loss, "f1_score": f1})

    # avvia la simulazione
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=lambda sr, arr: global_evaluate(sr, arr, testing=True),
    )

    # salvataggio pesi finali
    if context.run_config.get("save-weights", True):
        if param_name == "eta":
            param_str = f"_eta{eta}"
        elif param_name == "mu":
            param_str = f"_mu{mu}"
        else:
            param_str = ""

        weights_dir = str(Path(save_path) / "weights")
        os.makedirs(weights_dir, exist_ok=True)
        wt_seed_label = seed_label if seed_label else f"seed{seed}"
        filename = f"{strategy_name}{param_str}_etal{eta_l}_{wt_seed_label}.npz"
        np.savez(os.path.join(weights_dir, filename), *result.arrays.to_numpy_ndarrays())

    gc.collect()
