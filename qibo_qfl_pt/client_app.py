"""Quantum Federated Learning Client with Qiboml (PyTorch) and Flower."""





import gc
import numpy as np

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.client import ClientApp
from flwr.common import Context

from qibo_qfl_pt.task import (
    create_model, build_noise_model, build_client_config, train_model, evaluate_model,
    get_weights, set_weights, get_partition_id, load_data_client, set_seed,
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

    model_type = context.run_config.get("model-type", "quantum")

    noise_model, mitigation_config, nshots = build_client_config(context.run_config, partition_id)
    model = create_model(model_type=model_type, nshots=nshots, noise_model=noise_model, mitigation_config=mitigation_config)

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

    model_type = context.run_config.get("model-type", "quantum")

    noise_model, mitigation_config, nshots = build_client_config(context.run_config, partition_id)
    model = create_model(model_type=model_type, nshots=nshots, noise_model=noise_model, mitigation_config=mitigation_config)

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
    )

    loss, acc, f1 = evaluate_model(model, x_eval, y_eval)

    metrics = {"num-examples": len(x_eval), "loss": loss, "accuracy": acc, "f1": f1}

    content = RecordDict({
        "metrics": MetricRecord(metrics),
    })

    del model, x_eval, y_eval
    gc.collect()

    return Message(content=content, reply_to=msg)
