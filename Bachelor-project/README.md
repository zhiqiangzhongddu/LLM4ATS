# Bachelor-project
Pre-training Graph Neural Networks with Large Language Models.

To run the project go to /Pipeline and run this
```
python pipeline.py
```

To see the different arguments run
```
python pipeline.py --help
```

Properties are computed through either RDKit, Modred, or a combination of autodE and ORCA.
Molecules are by default represented as SMILES strings. RDKit and Mordred calculations use their own format, Mol, as input. For properties that are computed through either RDKit or Mordred, the list of molecules to compute from are first converted from SMILES to Mol by RDKit's built-in method Chem.MolFromSmiles(SMILES_STR). The molecules are kept as SMILES strings for properties that are computed through autodE and ORCA.

The selected properties and the computed values are written to specified files, if the respective arguments are given. The conversation with chat-GPT can be written to a file with '--save_conversation=True'. By default, the properties are randomly selected. If '--random_properties=False', the properties will be selected by chat-GPT.

Currently, the datasets are stored in /Data as txt files. The datasets are SMILES stringd seperated by \n. If you want to use your own dataset, specify so with '--MoleculeDataset'.

The list of properties/descriptors is stored in /Pipeline/descriptors.txt. NOTE: some of the RDKit calculators are deprecated, thus resulting in some of the properties being non-computable. These deprecated calculators are handled in RDKitDescriptors.py.