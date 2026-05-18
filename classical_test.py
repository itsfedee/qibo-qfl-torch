"""Test standalone: classical / quantum / quantum hybrid sul cerchio.

Cambia MODEL_TYPE in cima per scegliere quale architettura testare:
- "classical": MLP classica
- "quantum":   quantum puro (12 parametri)
- "hybrid":    quantum + pre-MLP (Linear(2,2) -> tanh -> quantum -> (z+1)/2)

LR_Q e LR_C permettono di settare LR diversi per parte quantum e classica.
"""

import torch
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float64)

# =====================================================================
# CONFIG -- modifica solo qui
# =====================================================================

MODEL_TYPE = "hybrid"    # "classical", "quantum", "hybrid"

# Comuni
SEED = 130
EPOCHS = 30
BATCH_SIZE = 8
N_TRAIN = 500
N_TEST = 200

# Learning rates
LR_C = 0.1               # LR per parametri classici (pre_net o MLP intera)
LR_Q = 0.1           # LR per parametri quantum

# Specifico classical
HIDDEN = 8                # hidden=3 -> 13 param, hidden=6 -> 25 param
PRE_HIDDEN = 4
POST_HIDDEN=4

# Specifico quantum
NQUBITS = 2
NLAYERS = 3

# =====================================================================
# Setup riproducibilità
# =====================================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

# =====================================================================
# Dati
# =====================================================================

rng = np.random.default_rng(SEED)
x_train_np = rng.uniform(-1, 1, (N_TRAIN, 2))
y_train_np = (np.linalg.norm(x_train_np, axis=1) <= 1.0).astype(np.float64)
x_test_np = rng.uniform(-1, 1, (N_TEST, 2))
y_test_np = (np.linalg.norm(x_test_np, axis=1) <= 1.0).astype(np.float64)

x_train = torch.tensor(x_train_np, dtype=torch.float64)
y_train = torch.tensor(y_train_np, dtype=torch.float64)
x_test = torch.tensor(x_test_np, dtype=torch.float64)
y_test = torch.tensor(y_test_np, dtype=torch.float64)

print(f"Train: {N_TRAIN} pts, positive fraction: {y_train.mean():.3f}")
print(f"Test:  {N_TEST} pts, positive fraction: {y_test.mean():.3f}\n")

# =====================================================================
# Modelli
# =====================================================================

class ClassicalModel(nn.Module):
    def __init__(self, hidden=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=torch.float64),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 1e-7, 1 - 1e-7)


def build_quantum_model():
    """Costruisce il QuantumModel di qiboml come nel task.py federated."""
    import qibo_qfl_pt.patches  # noqa: F401  -- applica patch PSR + CDR
    from qiboml.models.encoding import PhaseEncoding
    from qiboml.models.decoding import Expectation
    import qiboml.interfaces.pytorch as pt
    from qiboml.operations.differentiation import PSR
    from qibo import Circuit, gates, construct_backend
    from qibo.symbols import Z
    from qibo.hamiltonians import SymbolicHamiltonian

    encoding = PhaseEncoding(nqubits=NQUBITS)
    observable = SymbolicHamiltonian((Z(0) + Z(1)) / 2, nqubits=NQUBITS)
    decoding = Expectation(
        observable=observable,
        nqubits=NQUBITS,
        backend=construct_backend("numpy"),
        density_matrix=False,
        noise_model=None,
        nshots=None,
        mitigation_config=None,
    )

    def create_trainable_layer():
        circ = Circuit(NQUBITS)
        for q in range(NQUBITS):
            circ.add(gates.RX(q, theta=np.random.uniform(0, np.pi / 2)))
            circ.add(gates.RZ(q, theta=np.random.uniform(0, np.pi / 2)))
        for q in range(NQUBITS - 1):
            circ.add(gates.CNOT(q, q + 1))
        return circ

    circuit_structure = []
    for _ in range(NLAYERS):
        circuit_structure.extend([encoding, create_trainable_layer()])

    q_model = pt.QuantumModel(
        circuit_structure=circuit_structure,
        decoding=decoding,
        differentiation=PSR,
    )
    return q_model


class QMLModel(nn.Module):
    """Versione post-MLP: quantum -> MLP -> output."""
    def __init__(self, q_model, hybrid=False):
        super().__init__()
        self.q_model = q_model
        self.hybrid = hybrid
        if hybrid:
            self.post_net = nn.Sequential(
                nn.Linear(1, POST_HIDDEN, dtype=torch.float64),
                nn.Tanh(),
                nn.Linear(POST_HIDDEN, 1, dtype=torch.float64),
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


# =====================================================================
# Selezione modello + optimizer
# =====================================================================

if MODEL_TYPE == "classical":
    model = ClassicalModel(hidden=HIDDEN)
    label = f"ClassicalModel(hidden={HIDDEN})"
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_C)
    lr_str = f"LR={LR_C}"

elif MODEL_TYPE == "quantum":
    q = build_quantum_model()
    model = QMLModel(q, hybrid=False)
    label = f"QMLModel (pure quantum, {NQUBITS} qubits, {NLAYERS} layers)"
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_Q)
    lr_str = f"LR_Q={LR_Q}"

elif MODEL_TYPE == "hybrid":
    q = build_quantum_model()
    model = QMLModel(q, hybrid=True)
    label = f"QMLModel (hybrid pre-MLP, {NQUBITS} qubits, {NLAYERS} layers)"
    # Parameter groups: LR diverso per quantum e classico
    optimizer = torch.optim.SGD([
        {"params": model.q_model.parameters(), "lr": LR_Q},
        {"params": model.post_net.parameters(), "lr": LR_C},
    ])
    lr_str = f"LR_Q={LR_Q}, LR_C={LR_C}"

else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_q = sum(p.numel() for p in model.q_model.parameters()) if MODEL_TYPE != "classical" else 0
n_c = n_params - n_q

print(f"Model: {label}")
print(f"  Total params: {n_params}  (quantum: {n_q}, classical: {n_c})")
print(f"  {lr_str}, epochs={EPOCHS}, batch_size={BATCH_SIZE}\n")

# =====================================================================
# Training
# =====================================================================

loss_fn = nn.BCELoss()

print(f"{'epoch':>6} | {'train_loss':>10} | {'test_loss':>10} | {'test_acc':>8} | {'mean_pred':>10}")
print("-" * 60)

for epoch in range(EPOCHS):
    model.train()
    idx = torch.randperm(N_TRAIN)
    batch_losses = []
    for i in range(0, N_TRAIN, BATCH_SIZE):
        b = idx[i:i + BATCH_SIZE]
        optimizer.zero_grad()
        pred = model(x_train[b]).squeeze(-1)
        loss = loss_fn(pred, y_train[b])
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = float(np.mean(batch_losses))

    model.eval()
    with torch.no_grad():
        y_prob = model(x_test).squeeze(-1)
        test_loss = loss_fn(y_prob, y_test).item()
        preds = (y_prob >= 0.5).float()
        acc = (preds == y_test).float().mean().item()
        mean_pred = y_prob.mean().item()

  
    print(f"{epoch+1:>6} | {train_loss:>10.4f} | {test_loss:>10.4f} | {acc:>8.4f} | {mean_pred:>10.4f}")

print("\nFinal:")
print(f"  test_loss = {test_loss:.4f}")
print(f"  test_acc  = {acc:.4f}")
print(f"  mean_pred = {mean_pred:.4f}  (majority class ~ 0.78)")