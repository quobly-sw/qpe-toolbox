from dataclasses import dataclass


@dataclass(frozen=True)
class RegisterLayout:
    nbit_qpe: int
    nbit_aux: int
    nbit_sys: int

    @property
    def qpe(self) -> tuple[int, ...]:
        return tuple(range(self.nbit_qpe))

    @property
    def aux(self) -> tuple[int, ...]:
        return tuple(range(self.nbit_qpe, self.nbit_qpe + self.nbit_aux))

    @property
    def system(self) -> tuple[int, ...]:
        start = self.nbit_qpe + self.nbit_aux
        return tuple(range(start, start + self.nbit_sys))

    @property
    def aux_plus_system(self) -> tuple[int, ...]:
        return self.aux + self.system

    @property
    def total_qubits(self) -> int:
        return self.nbit_qpe + self.nbit_aux + self.nbit_sys
