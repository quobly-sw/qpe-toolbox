from dataclasses import dataclass
from functools import cached_property

import numpy as np
import quimb.tensor as qtn
from quimb.operator import HilbertSpace, SparseOperatorBuilder

from qpe_toolbox.tensor import kron_mpos


@dataclass(frozen=True)
class LCUData:
    terms: list[tuple[complex, str, list[int]]]

    def __post_init__(self):
        if len(self.terms) < 2:
            raise ValueError("LCU requires at least two terms.")
        for coeff, pauli, qubits in self.terms:
            if abs(coeff) < 1e-15:
                raise ValueError("Coefficient must be non-zero.")
            if not pauli:
                raise ValueError("Empty Pauli string.")
            if len(pauli) != len(qubits):
                raise ValueError(
                    "Pauli string length must match number of qubit indices."
                )
            if len(set(qubits)) != len(qubits):
                raise ValueError("Qubit indices in a term must be unique.")
            if any(q < 0 for q in qubits):
                raise ValueError("Qubit indices must be non-negative.")
            if not all(p in "ixyz" for p in pauli):
                raise ValueError(
                    "Pauli string contains invalid characters (allowed: i,x,y,z)."
                )

    @property
    def n_term(self) -> int:
        return len(self.terms)

    @property
    def alpha(self) -> list[float]:
        raw = [abs(c) for c, _, _ in self.terms]
        pad_len = 2**self.nbit_aux - self.n_term
        return raw + [0.0] * pad_len

    @property
    def norm1(self) -> float:
        return sum(abs(c) for c, _, _ in self.terms)

    @property
    def nbit_aux(self) -> int:
        return max(1, int(np.ceil(np.log2(self.n_term))))

    @property
    def nbit_sys(self) -> int:
        max_q = max(q for _, _, qlist in self.terms for q in qlist)
        return max_q + 1


class LCU:
    """
    Linear Combination of Unitaries (LCU) with lazy, cached operator construction.
    """

    def __init__(self, terms, cutoff=1e-10, max_bond=0):
        self.data = LCUData(terms)
        self.cutoff = cutoff
        self.max_bond = max_bond

    @classmethod
    def from_hamiltonian(cls, hamiltonian, cutoff=1e-10, max_bond=0):
        return cls(hamiltonian.terms, cutoff=cutoff, max_bond=max_bond)

    @property
    def terms(self):
        return self.data.terms

    @property
    def alpha(self):
        return self.data.alpha

    @property
    def norm1(self):
        return self.data.norm1

    @property
    def n_term(self):
        return self.data.n_term

    @property
    def nbit_aux(self):
        return self.data.nbit_aux

    @property
    def nbit_sys(self):
        return self.data.nbit_sys

    @cached_property
    def select_mpo(self, compress_every: int = 128):
        nbit_aux = self.nbit_aux
        nbit_sys = self.nbit_sys
        n_total = nbit_aux + nbit_sys

        def _new_builder():
            hilbert_space = HilbertSpace(range(n_total))
            return SparseOperatorBuilder(hilbert_space=hilbert_space)

        def _build_and_compress(builder):
            mpo = builder.build_mpo()
            mpo.compress(cutoff=self.cutoff, max_bond=self.max_bond)
            return mpo

        def _flush(builder, accumulated_mpo):
            """Build + compress the current chunk and merge into the accumulator."""
            chunk_mpo = _build_and_compress(builder)
            if accumulated_mpo is None:
                accumulated_mpo = chunk_mpo
            else:
                accumulated_mpo = accumulated_mpo + chunk_mpo
                accumulated_mpo.compress(cutoff=self.cutoff, max_bond=self.max_bond)
            return accumulated_mpo

        builder = _new_builder()
        accumulated_mpo = None
        n_in_builder = 0

        for l_term in range(2**nbit_aux):
            aux_ops = []
            for i in range(nbit_aux):
                bit = (l_term >> (nbit_aux - 1 - i)) & 1
                aux_ops.append(("h" if bit == 0 else "n", i))

            # ---- physical part ------------------------------------------------
            if l_term >= self.n_term:
                phys_ops = []
                phase = 1.0
            else:
                coeff, pauli_string, qubits = self.terms[l_term]
                if coeff == 0:
                    phys_ops = []
                    phase = 1.0
                else:
                    phys_ops = []
                    for q, p_raw in zip(qubits, pauli_string, strict=True):
                        p = p_raw.upper()
                        if p == "I":
                            continue
                        op_label = {"X": "x", "Y": "y", "Z": "z"}[p]
                        phys_ops.append((op_label, nbit_aux + q))
                    phase = coeff / abs(coeff)

            all_ops = aux_ops + phys_ops
            if all_ops:
                builder.add_term(phase, *all_ops)
                n_in_builder += 1

            # ---- intermediate build + compression ----------------------------
            if n_in_builder >= compress_every:
                accumulated_mpo = _flush(builder, accumulated_mpo)
                builder = _new_builder()
                n_in_builder = 0

        # ---- final partial chunk ---------------------------------------------
        if n_in_builder > 0:
            accumulated_mpo = _flush(builder, accumulated_mpo)

        mpo = accumulated_mpo
        print("select_mpo, max_bond = ", mpo.max_bond())
        return mpo

    @cached_property
    def prepare_mps(self):

        nbit_aux = self.nbit_aux
        alpha, norm1 = self.alpha, self.norm1
        mps_prep = np.sqrt(alpha[0] / norm1) * qtn.MPS_computational_state(
            "0" * nbit_aux
        )
        for i in range(1, self.n_term):
            mps_prep += np.sqrt(alpha[i] / norm1) * qtn.MPS_computational_state(
                f"{i:0{nbit_aux}b}"
            )
        mps_prep.compress(cutoff=self.cutoff, max_bond=self.max_bond)
        mps_prep /= mps_prep.norm()
        print("mps_prep, max_bond = ", mps_prep.max_bond())

        return mps_prep

    @cached_property
    def reflection_mpo(self):

        mps_prep = self.prepare_mps
        nbit_aux = self.nbit_aux
        L_mpo = mps_prep.partial_trace_to_mpo(keep=list(range(nbit_aux)))
        L_mpo.compress(cutoff=self.cutoff, max_bond=self.max_bond)

        nbit_sys = self.nbit_sys
        R_L = 2 * kron_mpos(L_mpo, qtn.MPO_identity(nbit_sys)) - qtn.MPO_identity(
            nbit_aux + nbit_sys
        )
        R_L.compress(cutoff=self.cutoff, max_bond=self.max_bond)
        print("R_L, max_bond = ", R_L.max_bond())

        return R_L

    @cached_property
    def walk_mpo(self):
        mpo = self.reflection_mpo.apply(
            self.select_mpo, compress=True, max_bond=self.max_bond
        )
        print("walk_mpo, max_bond = ", mpo.max_bond())
        return mpo
