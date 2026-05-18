"""Monkey-patches per qiboml e qibo."""

import numpy as np
import torch
import qibo.models.error_mitigation as _em
import qiboml.interfaces.pytorch as pt
from functools import reduce


# =====================================================================
# Patch 1: CDR deterministico
# sample_training_circuit_cdr chiama backend.set_seed(None) ad ogni
# iterazione, che resetta np.random con un seed casuale distruggendo
# la riproducibilità. Il fix installa una guardia temporanea che
# ignora le chiamate np.random.seed(None) durante l'esecuzione del CDR.
# =====================================================================

# Guardia permanente: impedisce a backend.set_seed(None) di randomizzare
# np.random. Questo è necessario perché diverse funzioni di qibo
# (sample_training_circuit_cdr, error_sensitive_circuit, ecc.) chiamano
# backend.set_seed(None) che resetta np.random con un seed casuale.
_original_np_seed = np.random.seed


def _guarded_seed(seed):
    """Ignora np.random.seed(None) per evitare reset casuali dell'RNG."""
    if seed is not None:
        _original_np_seed(seed)


np.random.seed = _guarded_seed

# Patch 1b: CDR isolato dal backend RNG
# CDR chiama np.random.seed(40) ma non resetta il backend RNG di qibo,
# che dipende dal seed esterno della run. Questo causa varianza tra run
# con seed diversi. Il fix resetta anche i backend di qibo prima di CDR.
_original_CDR = _em.CDR


def _isolated_CDR(*args, seed=None, **kwargs):
    if seed is not None:
        _original_np_seed(seed)
        _em.SIMULATION_BACKEND().set_seed(seed)
        _em.CLIFFORD_BACKEND().set_seed(seed)
    return _original_CDR(*args, seed=seed, **kwargs)


_em.CDR = _isolated_CDR


# =====================================================================
# Patch 2: QuantumModelAutoGrad fix
# Fix per il PSR che non gestisce correttamente i parametri.
# =====================================================================

def _get_angles(circuit, include_not_trainable):
    return np.array(
        [float(par)
         for params in circuit.get_parameters(include_not_trainable=include_not_trainable)
         for par in params],
        dtype=np.float32,
    )


class _FixedQuantumModelAutoGrad(torch.autograd.Function):

    # nel forward separo i parametri trainable da quelli totali (trainable + encoding)
    @staticmethod
    def forward(ctx, x, decoding, differentiation, circuit_tracer, *parameters):
        parameters = torch.stack(parameters)
        circuit, jacobian_wrt_inputs, jacobian, input_to_gate_map = circuit_tracer(
            parameters, x=x
        )
        dtype = getattr(decoding.backend.np, str(parameters.dtype).split(".")[-1])

        all_angles = decoding.backend.cast(_get_angles(circuit, True), dtype=dtype) # tutti
        for g, p in zip(differentiation.circuit.parametrized_gates, all_angles):
            g.parameters = p

        trainable_angles = decoding.backend.cast(_get_angles(circuit, False), dtype=dtype) # trainable

        ctx.save_for_backward(jacobian_wrt_inputs, jacobian)
        ctx.angles = trainable_angles # trainable
        ctx.all_angles = all_angles # tutti
        ctx.differentiation = differentiation
        ctx.input_to_gate_map = input_to_gate_map
        ctx.dtype = dtype
        ctx.wrt_inputs = jacobian_wrt_inputs is not None

        x_out = decoding(differentiation.circuit)
        del circuit
        x_out = torch.as_tensor(
            decoding.backend.to_numpy(x_out).tolist(),
            dtype=parameters.dtype,
            device=parameters.device,
        )
        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        jacobian_wrt_inputs, jacobian = ctx.saved_tensors
        backend = ctx.differentiation.decoding.backend

        # fix: ripristino lo stato del circuito per questo sample.
        # differentiation.circuit è condiviso: altri forward possono averne sovrascritto gli angoli
        # PSR usa lo stato corrente come baseline non-shiftato, quindi senza ripristino misurerebbe dal punto sbagliato.
   
        for g, p in zip(ctx.differentiation.circuit.parametrized_gates, ctx.all_angles):
            g.parameters = p

        angles_to_pass = ctx.all_angles if ctx.wrt_inputs else ctx.angles
        psr_result = ctx.differentiation.evaluate(angles_to_pass, wrt_inputs=ctx.wrt_inputs)
        jacobian_wrt_angles = torch.as_tensor(
            backend.to_numpy(psr_result),
            dtype=jacobian.dtype,
            device=jacobian.device,
        )
        del psr_result

        out_shape = ctx.differentiation.decoding.output_shape
        contraction = ((0, 1), (0,) + tuple(range(2, len(out_shape) + 2)))
        right_indices = tuple(range(1, len(grad_output.shape) + 1))
        left_indices = (0,) + right_indices

        if jacobian_wrt_inputs is not None:
            jacobian_wrt_encoding_angles = torch.vstack(
                [jacobian_wrt_angles[list(indices)]
                 for indices in zip(*ctx.input_to_gate_map.values())]
            )
            indices_to_discard = reduce(tuple.__add__, ctx.input_to_gate_map.values())
            jacobian_wrt_angles = torch.vstack(
                [row for i, row in enumerate(jacobian_wrt_angles)
                 if i not in indices_to_discard]
            ).reshape(-1, *out_shape)
            grad_input = torch.einsum(
                jacobian_wrt_inputs, contraction[0],
                jacobian_wrt_encoding_angles, contraction[1],
            )
            grad_input = torch.einsum(grad_input, left_indices, grad_output, right_indices)
        else:
            grad_input = None

        gradient = torch.einsum(
            jacobian, contraction[0], jacobian_wrt_angles, contraction[1]
        )
        gradient = torch.einsum(gradient, left_indices, grad_output, right_indices)
        return (grad_input, None, None, None, *gradient)


pt.QuantumModelAutoGrad = _FixedQuantumModelAutoGrad
