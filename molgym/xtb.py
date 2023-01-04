from re import search
from typing import List
from ase.calculators.calculator import FileIOCalculator
from ase.io import read


class XTB(FileIOCalculator):
    implemented_properties: List[str] = ['energy']
    command = "xtb mol.xyz --input calc.inp --ceasefiles 1> out.xot 2> /dev/null"
    discard_results_on_any_change = True

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='xtb', atoms=None, **kwargs):
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                                  atoms, **kwargs)
        self.calc = None

    def _input(self) -> None:
        with open(self.directory + '/calc.inp', 'w') as fd:
            lines = []
            for k, v in self.parameters.items():
                if isinstance(v, dict):
                    lines.append(f"${k}\n")
                    for sk, sv in v.items():
                        # Tabs do not work...
                        lines.append(f"  {sk}={sv}\n")
                    lines.append(f"$end\n")
                else:
                    lines.append(f"${k} {v}\n")
            fd.writelines(lines)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        atoms.write(self.directory + '/mol.xyz')
        self._input()
    
    def read_results(self):
        with open('out.xot') as fd:
            lines = fd.read()
        E = float(search(r"\| TOTAL ENERGY\s+(-?\d+\.\d+) Eh", lines)[1])
        self.results = dict(energy=E)

    def read(self, restart):
        self.atoms = read('mol.xyz')
