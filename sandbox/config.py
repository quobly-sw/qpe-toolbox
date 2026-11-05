from dataclasses import dataclass


@dataclass(frozen=True)
class QPEConfig:
    nbit_qpe: int
    n_shots: int = 0
    max_bond: int = 0
    cutoff: float = 1e-10
    verbosity: int = 1


@dataclass(frozen=True)
class RPEConfig:
    n_repetitions: int
    n_shots: int
    max_bond: int = 0
    cutoff: float = 1e-10
    verbosity: int = 1


@dataclass
class EstimateResult:
    theta: float
    folded_probabilities: dict[int, float]
    most_likely_state: int
