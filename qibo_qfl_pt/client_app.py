"""Quantum Federated Learning Client with Qiboml (PyTorch) and Flower."""

import gc
import numpy as np

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.client import ClientApp
from flwr.common import Context

from qibo_qfl_pt.task import (
    create_model, build_noise_model, build_client_config, train_model, evaluate_model,
    get_weights, set_weights, get_partition_id, load_data_client, set_seed, get_server_round
)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    seed = int(context.run_config["seed"])
    init_seed = int(context.run_config.get("init-seed", seed))
    data_seed = int(context.run_config.get("data-seed", seed))
    set_seed(seed)

    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose", True)
    lr = context.run_config["eta_l"]

    partition_id = get_partition_id(msg, context)
    server_round = get_server_round(msg, context)

    model_type = context.run_config.get("model-type", "quantum")
    hidden_classical = int(context.run_config.get("hidden-classical", 9))

    noise_model, mitigation_config, nshots = build_client_config(context.run_config, partition_id, server_round)
    model = create_model(model_type=model_type, nshots=nshots, noise_model=noise_model, mitigation_config=mitigation_config, hidden_classical=hidden_classical)

    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_weights(model, ndarrays)

    # Salva parametri globali per calcolo drift
    import torch
    global_params = [p.clone().detach() for p in model.parameters()]

    # FedProx: leggi mu dal config (le altre strategie non lo mandano → 0.0)
    proximal_mu = msg.content.get("config", {}).get("proximal-mu", 0.0)
    global_weights = [w.copy() for w in ndarrays] if proximal_mu > 0 else None

    # dati locali
    x_train, y_train = load_data_client(
        partition_id,
        ndata=context.run_config["n-train-data"],
        iid=context.run_config["iid"],
        num_partitions=context.run_config["num-clients"],
        alpha=context.run_config["alpha"],
        seed=data_seed,
    )

    # training
    history = train_model(
        model, x_train, y_train, lr=lr,
        epochs=epochs, batch_size=batch_size, verbose=verbose, partition_id=partition_id,
        global_weights=global_weights, proximal_mu=proximal_mu,
    )

    # metriche
    metrics = {"num-examples": len(x_train)}
    if history["loss"]:
        metrics["train_loss"] = history["loss"][-1]
    if history["accuracy"]:
        metrics["train_acc"] = history["accuracy"][-1]
    if history["f1"]:
        metrics["train_f1"] = history["f1"][-1]

    # Calcola drift: distanza L2 tra parametri locali e globali
    drift = sum(
        (p - gp).pow(2).sum() for p, gp in zip(model.parameters(), global_params)
    ).sqrt().item()
    metrics["drift"] = drift

    # Log della noise map REALMENTE USATA durante questo round di training.
    # E' questa (non quella ricostruita in eval) che con soglia alta resta "stale":
    # confrontandola con la mappa di eval si vede il disallineamento che frena la loss.
    import traceback
    try:
        mit = getattr(getattr(model, "q_model", None), "decoding", None)
        mit = getattr(mit, "mitigator", None)
        popt = getattr(mit, "_mitigation_map_popt", None) if mit is not None else None
        if popt is not None:
            a = popt[0].item() if hasattr(popt[0], "item") else float(popt[0])
            b = popt[1].item() if hasattr(popt[1], "item") else float(popt[1])
            metrics["cdr_a_train"] = float(a)
            metrics["cdr_b_train"] = float(b)
            metrics["cdr_client"] = int(partition_id)
            # numero di volte che la mappa e' stata (ri)fittata: con thr alta ~1, con thr bassa molte
            nmaps = getattr(mit, "_n_maps_computed", None)
            if nmaps is not None:
                metrics["cdr_n_maps"] = int(nmaps)
    except Exception:
        traceback.print_exc()

    # risposta al server
    content = RecordDict({
        "arrays": ArrayRecord(get_weights(model)),
        "metrics": MetricRecord(metrics),
    })

    del model, x_train, y_train, history
    gc.collect()

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    seed = int(context.run_config["seed"])
    init_seed = int(context.run_config.get("init-seed", seed))
    data_seed = int(context.run_config.get("data-seed", seed))
    set_seed(seed)

    partition_id = get_partition_id(msg, context)
    server_round = get_server_round(msg, context)

    model_type = context.run_config.get("model-type", "quantum")
    hidden_classical = int(context.run_config.get("hidden-classical", 9))

    noise_model, mitigation_config, nshots = build_client_config(context.run_config, partition_id, server_round)
    model = create_model(model_type=model_type, nshots=nshots, noise_model=noise_model, mitigation_config=mitigation_config, hidden_classical=hidden_classical)

    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_weights(model, ndarrays)

    x_eval, y_eval = load_data_client(
        partition_id,
        ndata=context.run_config["n-train-data"],
        iid=context.run_config["iid"],
        num_partitions=context.run_config["num-clients"],
        alpha=context.run_config["alpha"],
        seed=data_seed,
        client_eval=True,
        testing=context.run_config.get("testing", True),
    )

    loss, acc, f1 = evaluate_model(model, x_eval, y_eval)

    metrics = {"num-examples": len(x_eval), "loss": loss, "accuracy": acc, "f1": f1}

    # Log CDR (a, b) nelle metriche — NIENTE scrittura su file qui.
    # La scrittura concorrente su un JSON condiviso dai worker Ray paralleli
    # era la race condition che faceva fallire aggregate_evaluate dal round 3.
    # I parametri viaggiano nella MetricRecord e vengono persistiti dal server
    # (single-writer, round noto) insieme alle altre metriche di eval per-client.
    import traceback
    try:
        if hasattr(model, "q_model"):
            mitigator = getattr(model.q_model.decoding, "mitigator", None)
            popt = getattr(mitigator, "_mitigation_map_popt", None) if mitigator is not None else None
            if popt is not None:
                # popt puo' essere un tensore torch (backend PyTorch): usa .item().
                a = popt[0].item() if hasattr(popt[0], "item") else float(popt[0])
                b = popt[1].item() if hasattr(popt[1], "item") else float(popt[1])
                metrics["cdr_a"] = float(a)
                metrics["cdr_b"] = float(b)
                metrics["cdr_client"] = int(partition_id)

                # Scatter dei training point CDR (noisy vs noise-free), letto dallo
                # STESSO mitigator/fit di (a, b) -> retta e nuvola sempre coerenti.
                # MetricRecord accetta liste di scalari, quindi appiattiamo a float.
                td = getattr(mitigator, "_training_data", None)
                if td:
                    metrics["cdr_noisy"]     = [float(v) for v in td["noisy"]]
                    metrics["cdr_noisefree"] = [float(v) for v in td["noise-free"]]
    except Exception:
        # Il logging non deve MAI far fallire l'eval: stampa e prosegui.
        traceback.print_exc()

    content = RecordDict({
        "metrics": MetricRecord(metrics),
    })

    del model, x_eval, y_eval
    gc.collect()

    return Message(content=content, reply_to=msg)