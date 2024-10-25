from rdkit import Chem
from rdkit.Chem import (
    Descriptors, rdMolDescriptors, GraphDescriptors, Crippen,
    MolSurf, Fragments, AllChem
)
from rdkit.Chem.EState import EState_VSA
from rdkit.Chem.Descriptors3D import (
    Asphericity, Eccentricity, InertialShapeFactor, RadiusOfGyration
)


_rdkit_property_book = {
    # * Constitutional descriptors *#
    "Molecular Weight": {
        "Brief explanation": "Total mass of a molecule calculated as the sum of atomic weights of all atoms, providing fundamental information about molecular size and mass distribution.",
    },
    "Heavy Atom Molecular Weight": {
        "Brief explanation": "Sum of atomic weights of all non-hydrogen atoms in the molecule, useful for comparing core molecular frameworks.",
    },
    "Number of Atoms": {
        "Brief explanation": "Total count of all atoms in the molecule including hydrogens, providing basic structural information about molecular size.",
    },
    "Heavy Atom Count": {
        "Brief explanation": "Number of non-hydrogen atoms in the molecule, indicating the size of the molecular scaffold.",
    },
    "Number of Heteroatoms": {
        "Brief explanation": "Count of atoms that are neither carbon nor hydrogen (e.g., N, O, S, P), important for determining molecular polarity and reactivity.",
    },
    "Number of Valence Electrons": {
        "Brief explanation": "Total number of electrons in the outer shells of all atoms, crucial for understanding chemical bonding and reactivity patterns.",
    },
    "Total Formal Charge": {
        "Brief explanation": "Sum of all formal charges on atoms in the molecule, indicating overall molecular charge state and ionic character.",
    },

    # * Topological descriptors *#
    "Topological Polar Surface Area (TPSA)": {
        "Brief explanation": "Sum of surfaces of all polar atoms (mainly oxygen and nitrogen), correlating with drug absorption, including intestinal absorption and blood-brain barrier penetration.",
    },
    "Labute Approximate Surface Area (LabuteASA)": {
        "Brief explanation": "Approximate molecular surface area calculated using Labute's method, useful for predicting physical properties and molecular interactions.",
    },
    "Balaban J Index": {
        "Brief explanation": "Topological index based on molecular connectivity, indicating molecular branching and cyclicity. Higher values suggest more branched structures.",
    },
    "Bertz Complexity": {
        "Brief explanation": "Measure of molecular complexity considering both size and branching patterns. Higher values indicate more complex molecular structures.",
    },
    "Ipc (Information Content)": {
        "Brief explanation": "Graph-theoretical index measuring structural complexity based on the distribution of atomic neighborhoods in the molecular graph.",
    },
    "Chi0v": {
        "Brief explanation": "Zero-order valence connectivity index, reflecting atomic connectivity and valence state contributions to molecular structure.",
    },
    "Chi1v": {
        "Brief explanation": "First-order valence connectivity index, describing bond connectivity patterns and their contribution to molecular properties.",
    },
    "Chi2v": {
        "Brief explanation": "Second-order valence connectivity index, capturing larger structural fragments and their influence on molecular properties.",
    },

    # * Shape descriptors *#
    "Kappa Shape Index 1": {
        "Brief explanation": "First-order shape index reflecting the relative cyclicity of a molecule. Higher values indicate more linear structures.",
    },
    "Kappa Shape Index 2": {
        "Brief explanation": "Second-order shape index measuring spatial distribution of atoms. Sensitive to presence of rings and degree of branching.",
    },
    "Kappa Shape Index 3": {
        "Brief explanation": "Third-order shape index capturing more complex spatial arrangements. Particularly sensitive to centrally-located branching.",
    },

    # * Electronic properties *#
    "LogP": {
        "Brief explanation": "Logarithm of octanol-water partition coefficient, predicting molecular lipophilicity and membrane permeability. Key for drug absorption.",
    },
    "Molar Refractivity": {
        "Brief explanation": "Measure of total polarizability of a molecule, related to molecular volume and electronic properties. Important for predicting optical behavior.",
    },
    "EState VSA1": {
        "Brief explanation": "Sum of van der Waals surface areas of atoms with electrotopological state values in first range. Relates electronic state to molecular surface.",
    },
    "EState VSA2": {
        "Brief explanation": "Sum of van der Waals surface areas of atoms with electrotopological state values in second range. Describes different electronic environment contributions.",
    },
    "PEOE VSA1": {
        "Brief explanation": "Surface area contributions of atoms with specific partial charges calculated using Partial Equalization of Orbital Electronegativities method.",
    },
    "SlogP VSA1": {
        "Brief explanation": "Surface area contributions of atoms to lipophilicity (logP) in first range. Combines atomic logP with surface area.",
    },
    "SMR VSA1": {
        "Brief explanation": "Surface area contributions of atoms to molar refractivity in first range. Relates atomic refractivity to molecular surface.",
    },

    # * Hydrogen bonding *#
    "Number of Hydrogen Bond Donors": {
        "Brief explanation": "Count of NH and OH groups capable of donating hydrogen bonds. Critical for predicting molecular interactions and drug-like properties.",
    },
    "Number of Hydrogen Bond Acceptors": {
        "Brief explanation": "Count of oxygen and nitrogen atoms capable of accepting hydrogen bonds. Important for molecular recognition and solubility.",
    },
    "NHOH Group Count": {
        "Brief explanation": "Total number of nitrogen and oxygen atoms with attached hydrogens, indicating potential hydrogen bond donor sites.",
    },
    "NO Group Count": {
        "Brief explanation": "Total count of nitrogen and oxygen atoms, regardless of hydrogen attachment. Indicates overall polar atom content.",
    },

    # * Ring descriptors *#
    "Number of Rings": {
        "Brief explanation": "Total count of all rings in the molecular structure, including both aromatic and aliphatic rings. Key structural feature.",
    },
    "Number of Aromatic Rings": {
        "Brief explanation": "Count of rings with conjugated pi-electron systems exhibiting aromaticity. Important for molecular stability and interactions.",
    },
    "Aliphatic Rings Count": {
        "Brief explanation": "Number of non-aromatic (saturated or partially saturated) rings in the molecule. Affects molecular flexibility and properties.",
    },
    "Heterocycles Count": {
        "Brief explanation": "Number of rings containing at least one heteroatom (non-carbon). Important for biological activity and drug-like properties.",
    },

    # * Molecular flexibility *#
    "Number of Rotatable Bonds": {
        "Brief explanation": "Count of single bonds that can freely rotate, excluding those in rings. Key indicator of molecular flexibility and oral bioavailability.",
    },
    "Fraction of Csp3 Carbon Atoms": {
        "Brief explanation": "Ratio of sp3 hybridized carbons to total carbon count. Higher values indicate more 3D character and better drug-like properties.",
    },

    # * 3D descriptors *#
    "Molecular Volume in Å³": {
        "Brief explanation": "Three-dimensional space occupied by the molecule, calculated using van der Waals radii. Important for molecular packing and interactions.",
    },
    "Asphericity": {
        "Brief explanation": "Measure of how much molecular shape deviates from perfect sphere. Higher values indicate more elongated or irregular shapes.",
    },
    "Eccentricity": {
        "Brief explanation": "Ratio of the longest to shortest molecular axis. Indicates degree of molecular elongation in 3D space.",
    },
    "Radius of Gyration": {
        "Brief explanation": "Average distance of molecular mass from its center of mass. Describes molecular size and mass distribution in 3D.",
    },
    "Inertial Shape Factor": {
        "Brief explanation": "Ratio of principal moments of inertia, describing overall 3D shape and mass distribution. Distinguishes rod-like from disk-like molecules.",
    },

    # * Drug-likeness *#
    "QED": {
        "Brief explanation": "Quantitative Estimate of Drug-likeness, combining multiple molecular properties into a single score (0-1). Higher values indicate more drug-like characteristics.",
    }
}


# Dictionary to hold property names and calculation methods
_rdkit_property_tool = {
    # * Constitutional descriptors *#
    "Molecular Weight": {
        "method": Descriptors.ExactMolWt
    },
    "Heavy Atom Molecular Weight": {
        "method": Descriptors.HeavyAtomMolWt
    },
    "Number of Atoms": {
        "method": lambda m: m.GetNumAtoms()
    },
    "Heavy Atom Count": {
        "method": Descriptors.HeavyAtomCount
    },
    "Number of Heteroatoms": {
        "method": Descriptors.NumHeteroatoms
    },
    "Number of Valence Electrons": {
        "method": Descriptors.NumValenceElectrons
    },
    "Total Formal Charge": {
        "method": Chem.GetFormalCharge
    },

    # * Topological descriptors *#
    "Topological Polar Surface Area (TPSA)": {
        "method": MolSurf.TPSA
    },
    "Labute Approximate Surface Area (LabuteASA)": {
        "method": MolSurf.LabuteASA
    },
    "Balaban J Index": {
        "method": GraphDescriptors.BalabanJ
    },
    "Bertz Complexity": {
        "method": Descriptors.BertzCT
    },
    "Ipc (Information Content)": {
        "method": Descriptors.Ipc
    },
    "Chi0v": {
        "method": rdMolDescriptors.CalcChi0v
    },
    "Chi1v": {
        "method": rdMolDescriptors.CalcChi1v
    },
    "Chi2v": {
        "method": rdMolDescriptors.CalcChi2v
    },

    # * Shape descriptors *#
    "Kappa Shape Index 1": {
        "method": Descriptors.Kappa1
    },
    "Kappa Shape Index 2": {
        "method": Descriptors.Kappa2
    },
    "Kappa Shape Index 3": {
        "method": Descriptors.Kappa3
    },

    # * Electronic properties *#
    "LogP": {
        "method": Crippen.MolLogP
    },
    "Molar Refractivity": {
        "method": Crippen.MolMR
    },
    "EState VSA1": {
        "method": EState_VSA.EState_VSA1
    },
    "EState VSA2": {
        "method": EState_VSA.EState_VSA2
    },
    "PEOE VSA1": {
        "method": MolSurf.PEOE_VSA1
    },
    "SlogP VSA1": {
        "method": MolSurf.SlogP_VSA1
    },
    "SMR VSA1": {
        "method": MolSurf.SMR_VSA1
    },

    # * Hydrogen bonding *#
    "Number of Hydrogen Bond Donors": {
        "method": rdMolDescriptors.CalcNumHBD
    },
    "Number of Hydrogen Bond Acceptors": {
        "method": rdMolDescriptors.CalcNumHBA
    },
    "NHOH Group Count": {
        "method": Descriptors.NHOHCount
    },
    "NO Group Count": {
        "method": Descriptors.NOCount
    },

    # * Ring descriptors *#
    "Number of Rings": {
        "method": rdMolDescriptors.CalcNumRings
    },
    "Number of Aromatic Rings": {
        "method": rdMolDescriptors.CalcNumAromaticRings
    },
    "Aliphatic Rings Count": {
        "method": rdMolDescriptors.CalcNumAliphaticRings
    },
    "Heterocycles Count": {
        "method": rdMolDescriptors.CalcNumHeterocycles
    },

    # * Molecular flexibility *#
    "Number of Rotatable Bonds": {
        "method": rdMolDescriptors.CalcNumRotatableBonds
    },
    "Fraction of Csp3 Carbon Atoms": {
        "method": Descriptors.FractionCSP3
    },

    # * 3D descriptors *#
    "Molecular Volume in Å³": {
        "method": AllChem.ComputeMolVolume
    },
    "Asphericity": {
        "method": Asphericity
    },
    "Eccentricity": {
        "method": Eccentricity
    },
    "Radius of Gyration": {
        "method": RadiusOfGyration
    },
    "Inertial Shape Factor": {
        "method": InertialShapeFactor
    },

    # * Drug-likeness *#
    "QED": {
        "method": Descriptors.qed
    }
}


def get_available_fragments():
    """Get all available fragment descriptors from RDKit"""
    return {name: getattr(Fragments, name)
            for name in dir(Fragments)
            if name.startswith('fr_') and callable(getattr(Fragments, name))}

# Get all available fragment descriptors
fragment_descriptors = get_available_fragments()
fragment_property_book = {
    f"Fragment: {name[3:]}": {
        "Brief explanation": f"Count of {name[3:]} fragments"
    }
    for name, method in fragment_descriptors.items()
}
fragment_property_tool = {
    f"Fragment: {name[3:]}": {
        "method": method,
    }
    for name, method in fragment_descriptors.items()
}


# Add fragment properties
rdkit_property_book = _rdkit_property_book.copy()
rdkit_property_tool = _rdkit_property_tool.copy()
rdkit_property_book.update(fragment_property_book)
rdkit_property_tool.update(fragment_property_tool)


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
        # SMILES to mol
        mol = Chem.MolFromSmiles(smiles)

        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=42)
        if success != -1:
            AllChem.MMFFOptimizeMolecule(mol)

        print(f"\n=== {name} ===")
        for idx, prop in enumerate(rdkit_property_tool.keys()):
            brief_explanation = rdkit_property_book[prop]["Brief explanation"]
            try:
                value = rdkit_property_tool[prop]["method"](mol)
                if isinstance(value, float):
                    print(f"{idx} {prop}: {value:.4f}")
                    print(f"Brief explanation: {brief_explanation}")
                else:
                    print(f"{idx} {prop}: {value}")
                    print(f"Brief explanation: {brief_explanation}")
            except Exception as e:
                print(f"{idx} {prop}: Could not calculate - {str(e)}")
                print(f"Brief explanation: {brief_explanation}")
