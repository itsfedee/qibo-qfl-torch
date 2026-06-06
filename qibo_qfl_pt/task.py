"""Quantum Federated Learning with Qiboml (PyTorch) and Flower."""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import qibo
from qibo import Circuit, gates, set_backend, construct_backend, get_backend
from qibo.symbols import Z
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.noise import NoiseModel, PauliError, ReadoutError
from qiboml.models.encoding import PhaseEncoding
from qiboml.models.decoding import Expectation
import qiboml.interfaces.pytorch as pt
from qiboml.operations.differentiation import PSR

from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

qibo.config.log.setLevel("ERROR")
torch.set_default_dtype(torch.float64)
set_backend("numpy")

import qibo_qfl_pt.patches  


# =====================================================================
# Config
# =====================================================================

NQUBITS = 2
NLAYERS = 3 # testare anche con 6


# =====================================================================
# Seed
# =====================================================================

def set_seed(seed, backend=None):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if backend is not None:
        backend.set_seed(seed)
    else:
        qibo.get_backend().set_seed(seed)

    from qibo.models.error_mitigation import SIMULATION_BACKEND, CLIFFORD_BACKEND
    SIMULATION_BACKEND().set_seed(seed)
    CLIFFORD_BACKEND().set_seed(seed)



# =====================================================================
# Noise
# =====================================================================

def build_noise_model(pauli_base, readout_base, partition_id, scale):

    rng = np.random.default_rng(seed=partition_id)
    pauli_prob = float(np.clip(pauli_base + rng.uniform(-scale, scale), 0, 1))
    readout_prob = float(np.clip(readout_base + rng.uniform(-scale, scale), 0, 1))

    noise = NoiseModel()

    # local Pauli noise
    for q in range(NQUBITS):
        noise.add(
            PauliError([("X", pauli_prob), ("Y", pauli_prob), ("Z", pauli_prob)]),
            qubits=q,
        )

    # Readout noise
    single = np.array([
        [1 - readout_prob, readout_prob],
        [readout_prob, 1 - readout_prob],
    ])
    readout_matrix = np.kron(single, single)

    noise.add(
        ReadoutError(readout_matrix),
        gate=gates.M,
        qubits=[0, 1],
    )
    return noise, readout_prob



def build_client_config(run_config, partition_id):
    mode = run_config.get("mode", "noiseless")
    nshots = run_config.get("nshots", 1000)
    if nshots == "none":
        nshots = None
    else:
        nshots = int(nshots)

    if mode == "noiseless":
        return None, None, nshots

    noise_model, readout_prob = build_noise_model(
        pauli_base=run_config["base-pauli"],
        readout_base=run_config["base-readout"],
        partition_id=partition_id,
        scale=run_config["scale"],
    )
    
    mitigation_config = None
    if mode == "mitigated":
        single = np.array([[1 - readout_prob, readout_prob],
                           [readout_prob, 1 - readout_prob]])
        response_matrix = np.kron(single, single)
        mitigation_config = {
            "threshold": run_config.get("cdr-threshold", 0.01),
            "min_iterations": run_config.get("cdr-min-iterations", 500),
            "method": "CDR",
            "method_kwargs": {
                "n_training_samples": run_config.get("cdr-n-training-samples", 120),
                "nshots": run_config.get("cdr-nshots", 30000),
                "seed": run_config.get("cdr-seed", 40),
                "readout": {"response_matrix": response_matrix},
            },
        }
    return noise_model, mitigation_config, nshots


# =====================================================================
# Modello
# =====================================================================
"""
class QMLModel(nn.Module):
    def __init__(self, q_model, hybrid=False):
        super().__init__()
        self.q_model = q_model
        self.hybrid = hybrid

        if hybrid:
            self.post_net = nn.Sequential(
                nn.Linear(1, 4, dtype=torch.float64),
                nn.Tanh(),
                nn.Linear(4, 1, dtype=torch.float64),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x = torch.tanh(x) * torch.tensor(np.pi, dtype=torch.float64)
        x = torch.stack([self.q_model(xi).reshape(-1) for xi in x])
        if self.hybrid:
            x = x.double()
            x = self.post_net(x)
        else:
            x = (x + 1) / 2
        return torch.clamp(x, 1e-7, 1 - 1e-7)
"""
HIDDEN_HYBRID = 6
HIDDEN_CLASSICAL = 9


class ClassicalModel(nn.Module):
    def __init__(self, hidden=HIDDEN_CLASSICAL):
        super().__init__()
        self.hybrid = False
        self.net = nn.Sequential(
            nn.Linear(2, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=torch.float64),
            nn.Sigmoid(),
        )

    def forward(self, x):                           
        return torch.clamp(self.net(x), 1e-7, 1 - 1e-7)


class QMLModel(nn.Module):
    """Versione post-MLP: quantum -> MLP -> output."""
    def __init__(self, q_model, hybrid=False):
        super().__init__()
        self.q_model = q_model
        self.hybrid = hybrid
        if hybrid:
            self.post_net = nn.Sequential(
                nn.Linear(1, HIDDEN_HYBRID, dtype=torch.float64),
                nn.Tanh(),
                nn.Linear(HIDDEN_HYBRID, 1, dtype=torch.float64),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x = torch.tanh(x) * torch.tensor(np.pi, dtype=torch.float64)
        x = torch.stack([self.q_model(xi).reshape(-1) for xi in x])
        if self.hybrid:
            x = x.double()
            x = self.post_net(x)
        else:
            x = (x + 1) / 2
        return torch.clamp(x, 1e-7, 1 - 1e-7)


def create_model(model_type="quantum", noise_model=None, nshots=None, mitigation_config=None):

    if model_type == "classical":
        return ClassicalModel(hidden=HIDDEN_CLASSICAL)

    # quantum o hybrid
    encoding = PhaseEncoding(nqubits=NQUBITS)
    observable = SymbolicHamiltonian((Z(0) + Z(1)) / 2, nqubits=NQUBITS)
    decoding = Expectation(
        observable=observable,
        nqubits=NQUBITS,
        backend=construct_backend("numpy"),
        density_matrix=False if noise_model is None else True,
        noise_model=noise_model,
        nshots=nshots,
        mitigation_config=mitigation_config
    )

    circuit_structure = []
    for _ in range(NLAYERS):
        circuit_structure.extend([encoding, create_trainable_layer()])

    q_model = pt.QuantumModel(
        circuit_structure=circuit_structure,
        decoding=decoding,
        differentiation=PSR,
    )
    hybrid = (model_type == "hybrid")
    return QMLModel(q_model, hybrid=hybrid)


def create_trainable_layer():
    circ = Circuit(NQUBITS)
    for q in range(NQUBITS):
        circ.add(gates.RX(q, theta=np.random.uniform(0, np.pi / 2)))
        circ.add(gates.RZ(q, theta=np.random.uniform(0, np.pi / 2)))
    for q in range(NQUBITS - 1):
        circ.add(gates.CNOT(q, q + 1))
    return circ


# =====================================================================
# Pesi (per Flower)
# =====================================================================

def get_weights(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w, dtype=p.dtype, device=p.device)


# =====================================================================
# Training e valutazione
# =====================================================================


def train_model(model, x_train, y_train, epochs=5, lr=0.2, batch_size=16, verbose=True, partition_id=None,
                global_weights=None, proximal_mu=0.0):

    loss_fn = nn.BCELoss()

    if model.hybrid:
        param_groups = [
            {"params": list(model.q_model.parameters()), "lr": lr},
            {"params": list(model.post_net.parameters()), "lr": lr * 0.8},
        ]
    else:
        param_groups = [{"params": list(model.parameters()), "lr": lr}]
    optimizer = torch.optim.SGD(param_groups)

    history = {"loss": [], "accuracy": [], "f1": []}

    # Converti pesi globali in tensori (per FedProx)
    if proximal_mu > 0 and global_weights is not None:
        global_tensors = [torch.tensor(w, dtype=torch.float64) for w in global_weights]
    else:
        global_tensors = None

    for epoch in range(epochs):
        t0 = time.time()

        model.train()
        idx = torch.randperm(len(x_train))
        batch_losses = []
        for i in range(0, len(x_train), batch_size):
            b = idx[i : i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[b]).squeeze(-1).float(), y_train[b].float())
            if global_tensors is not None:
                prox_term = sum(
                    torch.sum((p - g) ** 2) for p, g in zip(model.parameters(), global_tensors)
                )
                loss = loss + (proximal_mu / 2) * prox_term
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        history["loss"].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            preds = (model(x_train).squeeze(-1) >= 0.5).float()
            acc = (preds == y_train).float().mean().item()
            f1 = f1_score(y_train.numpy(), preds.numpy(), average="macro", zero_division=0)
        history["accuracy"].append(acc)
        history["f1"].append(f1)

        t1 = time.time()
        if verbose:
            label = f"Client {partition_id}" if partition_id is not None else "Client"
            #print(f"  [{label}] Epoch {epoch+1}/{epochs}  loss: {epoch_loss:.4f}  acc: {acc:.4f}  f1: {f1:.4f}  time: {t1-t0:.2f}s")

    return history


def evaluate_model(model, x, y, verbose=False):

    loss_fn = nn.BCELoss()

    model.eval()
    with torch.no_grad():
        y_prob = model(x).squeeze(-1)
        loss = loss_fn(y_prob.float(), y.float()).item()
        preds = (y_prob >= 0.5).float()
        acc = (preds == y).float().mean().item()
        f1 = f1_score(y.numpy(), preds.numpy(), average="macro", zero_division=0)

    if verbose:
        print(f"  loss: {loss:.4f}  acc: {acc:.4f}  f1: {f1:.4f}")

    return loss, acc, f1


# =====================================================================
# Dataset
# =====================================================================

partitioner = None
last_seed = None


def generate_data(n_data, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n_data, 2))
    y = (np.linalg.norm(x, axis=1) <= 1.0).astype(np.int32)
    return {
        "point": x.astype(np.float64),
        "label": y.astype(np.int32),
    }


def load_data_server(ndata=500, testing=True, seed=40):
    current_seed = seed if testing else seed + 1
    data_dict = generate_data(ndata, current_seed)
    x = torch.tensor(data_dict["point"], dtype=torch.float64)
    y = torch.tensor(data_dict["label"].squeeze(), dtype=torch.float64)
    return x, y


def get_partition_id(msg, context):
    """Estrai il partition_id dal messaggio del server o dal config del nodo."""
    config_record = msg.content.get("config")
    if config_record and "partition_id" in config_record:
        return config_record["partition_id"]
    return context.node_config.get("partition-id", 0)


def load_data_client(partition_id, ndata=500, iid=True, num_partitions=5, alpha=0.25, seed=42, client_eval=False, testing=True):
    
    global partitioner, last_seed
    if partitioner is None or last_seed != seed:
        last_seed = seed
        train_dataset = Dataset.from_dict(generate_data(ndata, seed))
        if iid:
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                alpha=alpha,
                partition_by="label",
                seed=seed,
            )
        partitioner.dataset = train_dataset

    part = partitioner.load_partition(partition_id)
    part.set_format("numpy")
    x_train = torch.tensor(np.array(part["point"], dtype=np.float64), dtype=torch.float64)
    y_train = torch.tensor(np.array(part["label"], dtype=np.float64).squeeze(), dtype=torch.float64)

    if client_eval:
        eval_seed = partition_id + 1000 if testing else partition_id + 2000
        rng    = np.random.default_rng(seed=eval_seed)
        x_eval = torch.tensor(rng.uniform(-1, 1, size=(200, 2)), dtype=torch.float64)
        y_eval = torch.tensor((np.linalg.norm(x_eval.numpy(), axis=1) <= 1.0), dtype=torch.float64)
        return x_eval, y_eval
    return x_train, y_train