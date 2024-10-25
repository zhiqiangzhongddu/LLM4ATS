from rdkit import Chem
from rdkit.Chem import AllChem
from code.descriptors.rdkit_property import rdkit_property_book, rdkit_property_tool
from code.descriptors.physical_property import physical_molecular_property_book, PhysicalMolecularProperties


properties = (set(rdkit_property_book.keys())
              | set(physical_molecular_property_book.keys()))


class PropertyDescriptor:
    def __init__(self, property_name):
        assert property_name in properties, (
            "invalid property name {}".format(property_name))

        self.property_name = property_name

    def _calculate_one_property(self, smiles):

        if self.property_name in rdkit_property_tool.keys():
            # SMILES to mol
            mol = Chem.MolFromSmiles(smiles)

            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)
            success = AllChem.EmbedMolecule(mol, randomSeed=42)
            if success != -1:
                AllChem.MMFFOptimizeMolecule(mol)

            try:
                value = rdkit_property_tool[self.property_name]["method"](mol)
                if isinstance(value, float):
                    pass
                else:
                    value = None
            except Exception as e:
                value = None
                print(f"{self.property_name}: Could not calculate - {str(e)}")

            return value

        elif self.property_name in physical_molecular_property_book.keys():
            mol = PhysicalMolecularProperties(smiles)
            properties = mol.get_all_properties()
            try:
                value = properties[self.property_name]
            except KeyError:
                print(f"\nProperty name: {self.property_name}")
                for item in properties:
                    print(item)

            return value

        else:
            raise ValueError('Invalid Property Name')

    def calculate_property(self, mol):

        if isinstance(mol, list):
            return [self._calculate_one_property(smiles=smiles) for smiles in mol]
        else:
            return self._calculate_one_property(smiles=mol)


if __name__ == '__main__':
    from code.data_utils.dataset import DatasetLoader

    for prop in properties:
        descriptor = PropertyDescriptor(property_name=prop)
        dataset = DatasetLoader()
        _, text = dataset.load_data()
        smiles = [text[42], text[420]]

        value = descriptor.calculate_property(mol=smiles)
        print(prop, value)