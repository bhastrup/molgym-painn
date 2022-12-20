import abc
import logging
import time
from typing import Tuple, Dict

import numpy as np
from ase import Atoms, Atom

from molgym.xtb import XTB


class MolecularReward(abc.ABC):
    @abc.abstractmethod
    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        raise NotImplementedError

    @staticmethod
    def get_minimum_spin_multiplicity(atoms: Atoms) -> int:
        return atoms.numbers.sum() % 2 + 1
    
    @staticmethod
    def get_spin(atoms) -> int:
        return MolecularReward.get_minimum_spin_multiplicity(atoms) - 1


class InteractionReward(MolecularReward):
    def __init__(self) -> None:
        self.calculator = XTB()

        self.settings = {
            'chrg': 0,
            'scc': {'maxiterations': 196},
        }

        self.atom_energies: Dict[str, float] = {}

    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        start = time.time()

        all_atoms = atoms.copy()
        all_atoms.append(new_atom)

        e_tot = self._calculate_energy(all_atoms)
        e_parts = self._calculate_energy(atoms) + self._calculate_atomic_energy(new_atom)
        delta_e = e_tot - e_parts
        logging.debug(f'e_tot: {e_tot},   delta_e: {delta_e}')
        elapsed = time.time() - start

        reward = -1 * delta_e

        info = {
            'elapsed_time': elapsed,
        }

        if e_tot == 1234.0:
            return -1234.0, info

        return reward, info

    def _calculate_atomic_energy(self, atom: Atom) -> float:
        if atom.symbol not in self.atom_energies:
            atoms = Atoms()
            atoms.append(atom)
            self.atom_energies[atom.symbol] = self._calculate_energy(atoms)
        return self.atom_energies[atom.symbol]

    def _calculate_energy(self, atoms: Atoms) -> float:
        if len(atoms) == 0:
            return 0.0
        self.calculator = XTB()
        self.settings['spin'] = self.get_spin(atoms)
        self.calculator.set(**self.settings)
        self.calculator.atoms = atoms
        try:
            energy = self.calculator.get_potential_energy()
        except:
            logging.warning('XTB calculation has FAILED!')
            energy = 1234.0
        return energy


class SolvationReward(InteractionReward):
    def __init__(self, distance_penalty=0.01) -> None:
        super().__init__()

        self.distance_penalty = distance_penalty

    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        start_time = time.time()
        self.calculator = XTB()

        all_atoms = atoms.copy()
        all_atoms.append(new_atom)

        e_tot = self._calculate_energy(all_atoms)
        e_parts = self._calculate_energy(atoms) + self._calculate_atomic_energy(new_atom)
        delta_e = e_tot - e_parts

        distance = np.linalg.norm(new_atom.position)

        reward = -1 * (delta_e + self.distance_penalty * distance)

        info = {
            'elapsed_time': time.time() - start_time,
        }

        return reward, info
