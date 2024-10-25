from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from scipy.constants import R, h, c, N_A, k, epsilon_0
import math


physical_molecular_property_book = {
    "Dipole Moment": {
        # "method":  "calculate_dipole_moment",
        "Brief explanation": "Measure of charge separation in a molecule, calculated from atomic partial charges and their 3D positions. Indicates polarity and is measured in Debye units.",
    },
    "Isotropic Polarizability": {
        # "method":  "calculate_isotropic_polarizability",
        "Brief explanation": "Measure of how easily electron density can be distorted by an external electric field, calculated from atomic contributions and bond effects. Given in cubic Angstroms.",
    },
    "HOMO Energy": {
        # "method":  "calculate_orbital_energies",
        "Brief explanation": "Energy level of the highest filled molecular orbital, determines the molecule's ionization potential and electron-donating tendency in chemical reactions.",
    },
    "LUMO Energy": {
        # "method":  "calculate_orbital_energies",
        "Brief explanation": "Energy level of the lowest unfilled molecular orbital, determines the molecule's electron affinity and electron-accepting tendency in chemical reactions.",
    },
    "HOMO-LUMO Gap": {
        # "method":  "calculate_orbital_energies",
        "Brief explanation": "Energy difference between highest occupied and lowest unoccupied molecular orbitals. Indicates molecular stability and reactivity - smaller gaps suggest higher reactivity.",
    },
    "Electronic Spatial Extent": {
        # "method":  "calculate_electronic_spatial_extent",
        "Brief explanation": "Measure of molecular size based on electronic distribution, calculated from atomic positions weighted by nuclear charges. Given in atomic units.",
    },
    "Zero Point Vibrational Energy": {
        # "method":  "calculate_vibrational_properties",
        "Brief explanation": "Lowest possible vibrational energy of a molecule at absolute zero temperature, calculated from estimated vibrational frequencies.",
    },
    "Internal Energy at 0K": {
        # "method":  "calculate_thermodynamic_properties",
        "Brief explanation": "Total molecular energy at absolute zero, including electronic, vibrational, and nuclear contributions. Base reference for energy calculations.",
    },
    "Internal Energy at 298.15K": {
        # "method":  "calculate_thermodynamic_properties",
        "Brief explanation": "Total molecular energy at room temperature (298.15K), including thermal energy contributions from vibrations, rotations, and translations.",
    },
    "Enthalpy at 298.15K": {
        # "method":  "calculate_thermodynamic_properties",
        "Brief explanation": "Internal energy plus pressure-volume work (PV) at room temperature. Represents total heat content of the molecule.",
    },
    "Free Energy at 298.15K": {
        # "method":  "calculate_thermodynamic_properties",
        "Brief explanation": "Gibbs free energy at room temperature, combining enthalpy and entropy. Determines spontaneity of chemical processes.",
    },
    "Heat Capacity at 298.15K": {
        # "method":  "calculate_thermodynamic_properties",
        "Brief explanation": "Amount of heat required to raise molecular temperature by one degree at constant volume, including vibrational, rotational, and translational contributions.",
    },
    "Atomization Energy at 0K": {
        # "method":  "calculate_atomization_energies",
        "Brief explanation": "Energy required to break molecule into isolated atoms at 0K, calculated as difference between atomic reference energies and molecular energy.",
    },
    "Atomization Energy at 298.15K": {
        # "method":  "calculate_atomization_energies",
        "Brief explanation": "Energy required to break molecule into isolated atoms at room temperature, including thermal energy contributions.",
    },
    "Atomization Enthalpy at 298.15K": {
        # "method":  "calculate_atomization_energies",
        "Brief explanation": "Heat absorbed when breaking molecule into isolated atoms at room temperature, including PV work.",
    },
    "Atomization Free Energy at 298.15K": {
        # "method":  "calculate_atomization_energies",
        "Brief explanation": "Gibbs free energy change for breaking molecule into isolated atoms at room temperature, including entropy effects.",
    },
}


class PhysicalMolecularProperties:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)

        if self.mol is None:
            raise ValueError("Invalid SMILES string")

        self.mol = Chem.AddHs(self.mol)
        success = AllChem.EmbedMolecule(self.mol, randomSeed=42)
        if success == -1:
            raise ValueError("Could not generate 3D coordinates")

        AllChem.MMFFOptimizeMolecule(self.mol)

        self.num_atoms = self.mol.GetNumAtoms()
        self.num_electrons = sum(atom.GetAtomicNum() for atom in self.mol.GetAtoms())
        self.electronic_structure = self._calculate_electronic_structure()

        # Reference atomic energies in kJ/mol for atomization energy calculations
        self.atomic_energies = {
            'H': -1312.0,
            'C': -37829.0,
            'N': -54714.0,
            'O': -75065.0,
            'F': -99718.0,
            'Cl': -460146.0,
            'Br': -2559994.0,
            'I': -6917947.0,
            'S': -397648.0,
            'P': -341259.0
        }

    def _calculate_electronic_structure(self):
        """Calculate approximate electronic structure using extended Hückel-like approach."""
        # Orbital energy parameters (in eV)
        orbital_energies = {
            'C': {'2s': -21.4, '2p': -11.4},
            'N': {'2s': -26.0, '2p': -13.4},
            'O': {'2s': -32.3, '2p': -14.8},
            'H': {'1s': -13.6},
            'F': {'2s': -40.0, '2p': -18.1},
            'Cl': {'3s': -30.0, '3p': -15.0},
            'Br': {'4s': -27.0, '4p': -13.1},
            'I': {'5s': -23.0, '5p': -12.7},
            'S': {'3s': -20.0, '3p': -13.3},
            'P': {'3s': -18.6, '3p': -14.0}
        }

        # Hybridization correction factors
        hybridization_corrections = {
            Chem.HybridizationType.SP3: 0.5,
            Chem.HybridizationType.SP2: 1.0,
            Chem.HybridizationType.SP: 1.5
        }

        orbital_list = []
        total_electrons = 0
        bond_orders = []

        for atom in self.mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in orbital_energies:
                continue

            # Count valence electrons
            valence = atom.GetTotalValence()
            total_electrons += valence

            # Get hybridization correction
            hybridization = atom.GetHybridization()
            hyb_correction = hybridization_corrections.get(hybridization, 0.0)

            # Calculate total bond order
            total_bond_order = sum(bond.GetBondTypeAsDouble()
                                   for bond in atom.GetBonds())
            bond_orders.append(total_bond_order)

            # Add orbital energies with corrections
            for orbital, base_energy in orbital_energies[symbol].items():
                # Apply hybridization correction
                energy = base_energy + hyb_correction

                # Apply bond order correction
                energy += 0.2 * total_bond_order

                # Apply electronegativity correction
                energy += 0.1 * atom.GetExplicitValence()

                orbital_list.append(energy)

        # Sort orbital energies
        orbital_list.sort(reverse=True)

        # Calculate HOMO index (assuming closed shell)
        homo_index = (total_electrons // 2) - 1 if orbital_list else -1

        return {
            'orbital_energies': orbital_list,
            'homo_index': homo_index,
            'total_electrons': total_electrons,
            'bond_orders': bond_orders
        }

    def calculate_orbital_energies(self):
        """Calculate HOMO/LUMO energies and gap."""
        orbitals = self.electronic_structure['orbital_energies']
        homo_idx = self.electronic_structure['homo_index']

        if not orbitals or homo_idx < 0 or homo_idx >= len(orbitals):
            return {
                'HOMO Energy': None,
                'LUMO Energy': None,
                'HOMO-LUMO Gap': None
            }

        homo = orbitals[homo_idx]
        lumo = orbitals[homo_idx + 1] if homo_idx + 1 < len(orbitals) else None
        gap = lumo - homo if lumo is not None else None

        return {
            'HOMO Energy': homo,
            'LUMO Energy': lumo,
            'HOMO-LUMO Gap': gap
        }

    def calculate_isotropic_polarizability(self):
        """Calculate molecular polarizability using atomic contributions and bond effects."""
        # Atomic polarizabilities in Å³
        atomic_polarizabilities = {
            'H': 0.387,
            'C': 1.76,
            'N': 1.10,
            'O': 0.802,
            'F': 0.557,
            'Cl': 2.18,
            'Br': 3.05,
            'I': 5.16,
            'S': 2.90,
            'P': 1.82
        }

        # Bond contribution factors
        bond_factors = {
            Chem.BondType.SINGLE: 0.05,
            Chem.BondType.DOUBLE: 0.15,
            Chem.BondType.TRIPLE: 0.25,
            Chem.BondType.AROMATIC: 0.10
        }

        # Calculate base polarizability from atomic contributions
        total_polarizability = 0.0
        for atom in self.mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atomic_polarizabilities:
                total_polarizability += atomic_polarizabilities[symbol]

        # Add bond contributions
        for bond in self.mol.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type in bond_factors:
                total_polarizability += bond_factors[bond_type]

        return total_polarizability

    def calculate_electronic_spatial_extent(self):
        """Calculate electronic spatial extent."""
        conf = self.mol.GetConformer()
        positions = []
        weights = []

        for i in range(self.num_atoms):
            pos = conf.GetAtomPosition(i)
            atom = self.mol.GetAtomWithIdx(i)
            positions.append([pos.x, pos.y, pos.z])
            weights.append(atom.GetAtomicNum())

        positions = np.array(positions)
        weights = np.array(weights)

        # Calculate center of nuclear charge
        center = np.average(positions, weights=weights, axis=0)

        # Calculate electronic spatial extent
        r_squared = 0
        total_weight = sum(weights)

        for pos, weight in zip(positions, weights):
            r_squared += weight * np.sum((pos - center) ** 2)

        return r_squared / total_weight if total_weight > 0 else 0

    def calculate_dipole_moment(self):
        """Calculate dipole moment magnitude and vector."""
        try:
            conf = self.mol.GetConformer()
            props = AllChem.MMFFGetMoleculeProperties(self.mol)

            if props is None:
                return None

            charges = [props.GetMMFFPartialCharge(i) for i in range(self.num_atoms)]

            # Calculate dipole moment components
            dx = dy = dz = 0.0
            for i, charge in enumerate(charges):
                pos = conf.GetAtomPosition(i)
                dx += charge * pos.x
                dy += charge * pos.y
                dz += charge * pos.z

            # Convert to Debye (1 e⋅Å ≈ 4.803 Debye)
            debye_conversion = 4.803
            dipole_vector = (dx * debye_conversion,
                             dy * debye_conversion,
                             dz * debye_conversion)
            magnitude = np.sqrt(sum(d * d for d in dipole_vector))

            return magnitude, dipole_vector

        except Exception as e:
            print(f"Warning: Dipole moment calculation failed - {str(e)}")
            return None

    def calculate_vibrational_properties(self):
        """Calculate vibrational properties including zero-point energy."""

        def estimate_frequencies():
            frequencies = []

            # Calculate frequencies for each bond
            for bond in self.mol.GetBonds():
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()

                # Force constants in mdyne/Å
                force_constants = {
                    Chem.BondType.SINGLE: 5.0,
                    Chem.BondType.DOUBLE: 10.0,
                    Chem.BondType.TRIPLE: 15.0,
                    Chem.BondType.AROMATIC: 6.5
                }

                k = force_constants.get(bond.GetBondType(), 5.0)
                k *= 100  # Convert to N/m

                # Calculate reduced mass
                m1 = atom1.GetMass() * 1.66053904e-27  # amu to kg
                m2 = atom2.GetMass() * 1.66053904e-27
                reduced_mass = (m1 * m2) / (m1 + m2)

                # Calculate frequency in cm^-1
                freq = math.sqrt(k / reduced_mass) / (2 * math.pi * c * 100)
                frequencies.append(freq)

            # Add bending modes
            n_bending = max(0, 3 * self.num_atoms - 6 - len(frequencies))
            frequencies.extend([1000] * n_bending)

            return frequencies

        frequencies = estimate_frequencies()

        # Calculate Zero Point Vibrational Energy
        if frequencies:
            zpe = sum(0.5 * h * f * c * 100 * N_A for f in frequencies)  # J/mol
            return {
                'Zero Point Vibrational Energy': zpe / 1000,
                'Vibrational Frequencies': frequencies
            }
        else:
            return {
                'Zero Point Vibrational Energy': 0.0,
                'Vibrational Frequencies': []
            }

    def calculate_thermodynamic_properties(self):
        """Calculate thermodynamic properties."""
        vib_props = self.calculate_vibrational_properties()
        frequencies = vib_props['Vibrational Frequencies']

        if not frequencies:
            return {
                'Internal Energy at 0K': 0.0,
                'Internal Energy at 298.15K': 0.0,
                'Enthalpy at 298.15K': 0.0,
                'Free Energy at 298.15K': 0.0,
                'Heat Capacity at 298.15K': 0.0
            }

        T = 298.15  # Standard temperature

        # Calculate vibrational contributions
        E_vib = S_vib = Cv_vib = 0.0
        for freq in frequencies:
            x = h * freq * c * 100 / (k * T)
            if x > 0:
                E_vib += R * T * x / (2 * (1 - math.exp(-x)))
                S_vib += R * (x / (math.exp(x) - 1) - math.log(1 - math.exp(-x)))
                Cv_vib += R * x ** 2 * math.exp(x) / (1 - math.exp(x)) ** 2

        # Add translational and rotational contributions
        E_trans = 3 * R * T / 2
        E_rot = 3 * R * T / 2

        # Calculate total internal energy
        U_0 = vib_props['Zero Point Vibrational Energy'] * 1000  # Convert to J/mol
        U_298 = U_0 + E_vib + E_trans + E_rot

        # Calculate enthalpy (H = U + PV = U + RT for ideal gas)
        H_298 = U_298 + R * T

        # Calculate entropy contributions
        S_trans = R * (5 / 2 + math.log((2 * math.pi * self.num_atoms * k * T) ** (3 / 2)))
        S_rot = R * (3 / 2 + math.log((8 * math.pi ** 2 * k * T) ** (3 / 2)))
        S_total = S_vib + S_trans + S_rot

        # Calculate Gibbs free energy
        G_298 = H_298 - T * S_total

        return {
            'Internal Energy at 0K': U_0 / 1000,
            'Internal Energy at 298.15K': U_298 / 1000,
            'Enthalpy at 298.15K': H_298 / 1000,
            'Free Energy at 298.15K': G_298 / 1000,
            'Heat Capacity at 298.15K': Cv_vib + 3 * R
        }

    def calculate_atomization_energies(self):
        """Calculate atomization energies at different conditions."""
        # Get molecular energies
        thermo = self.calculate_thermodynamic_properties()
        molecular_energy_0K = thermo['Internal Energy at 0K']
        molecular_energy_298K = thermo['Internal Energy at 298.15K']
        molecular_enthalpy_298K = thermo['Enthalpy at 298.15K']
        molecular_free_energy_298K = thermo['Free Energy at 298.15K']

        # Calculate total atomic reference energy
        total_atomic_energy = 0.0
        for atom in self.mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in self.atomic_energies:
                total_atomic_energy += self.atomic_energies[symbol]

        # Calculate atomization energies
        return {
            'Atomization Energy at 0K': total_atomic_energy - molecular_energy_0K,
            'Atomization Energy at 298.15K': total_atomic_energy - molecular_energy_298K,
            'Atomization Enthalpy at 298.15K': total_atomic_energy - molecular_enthalpy_298K,
            'Atomization Free Energy at 298.15K': total_atomic_energy - molecular_free_energy_298K
        }

    def get_all_properties(self):
        """Get all molecular properties."""
        properties = {}

        # Electronic properties
        orbital_energies = self.calculate_orbital_energies()
        properties.update({
            'HOMO Energy': orbital_energies['HOMO Energy'],
            'LUMO Energy': orbital_energies['LUMO Energy'],
            'HOMO-LUMO Gap': orbital_energies['HOMO-LUMO Gap']
        })

        # Dipole moment
        dipole_result = self.calculate_dipole_moment()
        if dipole_result is not None:
            magnitude, vector = dipole_result
            properties['Dipole Moment'] = magnitude
            properties['Dipole Vector'] = vector

        # Polarizability
        properties['Isotropic Polarizability'] = self.calculate_isotropic_polarizability()

        # Electronic spatial extent
        properties['Electronic Spatial Extent'] = self.calculate_electronic_spatial_extent()

        # Vibrational and thermodynamic properties
        vib_props = self.calculate_vibrational_properties()
        properties['Zero Point Vibrational Energy'] = vib_props['Zero Point Vibrational Energy']

        thermo_props = self.calculate_thermodynamic_properties()
        properties.update(thermo_props)

        # Atomization energies
        atomization_props = self.calculate_atomization_energies()
        properties.update(atomization_props)

        return properties


if __name__ == "__main__":
    examples = {
        'Water': 'O',
        'Methane': 'C',
        'Ethanol': 'CCO',
        'Benzene': 'c1ccccc1',
        'Acetone': 'CC(=O)C',
        'Ethene': 'C=C',
        'Propane': 'CCC'
    }

    for name, smiles in examples.items():
        print(f"\n=== {name} ===")
        try:
            mol = PhysicalMolecularProperties(smiles)
            properties = mol.get_all_properties()

            print(f"Analysis for molecule: {smiles}")
            print("-" * 50)
            for prop, value in properties.items():
                if isinstance(value, (int, float)):
                    print(f"{prop}: {value:.3f}")
                else:
                    print(f"{prop}: {value}")

        except Exception as e:
            print(f"Error analyzing molecule: {str(e)}")
