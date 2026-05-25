"""Diagnostica RNG: identifica quale generatore causa il fallimento CDR per seed=4."""

import random
import numpy as np
import torch
import qibo
from pathlib import Path

from qibo_qfl_pt.task import (
    create_model, build_noise_model, set_weights, set_seed, load_data_server, evaluate_model
)

# -------- carica pesi seed=4 una volta sola --------
WEIGHTS_PATH = "results/iid/fedavg/mitigated/scaled_p0.003_r0.003_s0.002/nshots_1000/weights/FedAvg_etal0.3_seed4.npz"  
data = np.load(WEIGHTS_PATH)
weights_seed4 = [data[k] for k in data.files]

# -------- dati di valutazione fissi (indipendenti dal seed esperimento) --------
x_eval, y_eval = load_data_server(ndata=200, testing=True)  # usa seed interno fisso=40

# -------- noise model fisso (partition_id=0, indipendente dal seed) --------
noise_model, readout_prob = build_noise_model(
    pauli_base=0.005, readout_base=0.005, partition_id=0, scale=0.002,
)
single = np.array([[1 - readout_prob, readout_prob],
                   [readout_prob, 1 - readout_prob]])
response_matrix = np.kron(single, single)
mitigation_config = {
    "threshold": 0.0,
    "min_iterations": 500,
    "method": "CDR",
    "method_kwargs": {
        "n_training_samples": 150,
        "nshots": 40000,
        "seed": 40,
        "readout": {"response_matrix": response_matrix},
    },
}


def run_forward():
    """Costruisce modello mitigato con pesi seed4, fa forward, ritorna loss."""
    model = create_model(
        nshots=1000,
        noise_model=noise_model,
        mitigation_config=mitigation_config,
    )
    set_weights(model, weights_seed4)
    loss, acc, f1 = evaluate_model(model, x_eval, y_eval)
    return loss


# ============================================================
# I 4 TEST
# ============================================================

NEUTRAL = 99  # un seed "benigno" da quanto hai visto

print("="*60)

# baseline: tutto a stato seed=4 (patologico)
set_seed(4)
loss_baseline = run_forward()
print(f"baseline (tutto a 4):              loss={loss_baseline:.4f}")

# test A: solo random (Python standard)
set_seed(4)
random.seed(NEUTRAL)
loss_A = run_forward()
print(f"A) random.seed({NEUTRAL}) prima:           loss={loss_A:.4f}")

# test B: solo torch
set_seed(4)
torch.manual_seed(NEUTRAL)
loss_B = run_forward()
print(f"B) torch.manual_seed({NEUTRAL}) prima:     loss={loss_B:.4f}")

# test C: solo qibo backend
set_seed(4)
qibo.get_backend().set_seed(NEUTRAL)
loss_C = run_forward()
print(f"C) qibo backend set_seed({NEUTRAL}) prima: loss={loss_C:.4f}")

# test D: solo np.random
set_seed(4)
np.random.seed(NEUTRAL)
loss_D = run_forward()
print(f"D) np.random.seed({NEUTRAL}) prima:        loss={loss_D:.4f}")

# riferimento noiseless per confronto
model_nl = create_model(nshots=1000, noise_model=None, mitigation_config=None)
set_weights(model_nl, weights_seed4)
loss_nl, _, _ = evaluate_model(model_nl, x_eval, y_eval)
print(f"\nriferimento noiseless:             loss={loss_nl:.4f}")
print("="*60)
print("Il colpevole è il test che fa scendere la loss vicino a noiseless.")