# Bachelor-project
Pre-training Graph Neural Networks with Large Language Models.

To run the project run this
```
python Pipeline/pipeline.py
```

To see the different arguments run
```
python Pipeline/pipeline.py --help
```

By default, the properties are selected by the LLM. If '--random_properties' is set, the properties will be selected randomly. NOTE: It is possible to use previously selected properties by setting the '--use_prev_llm_props' flag. Otherwise the LLM will be asked to select new properties. The results of previous settings are stored in /Pipeline/LLMqueries/txt_files/counters. The conversation with chat-GPT can be written to a file with '--save_conversation'.

Currently, the datasets are stored in /Pipeline/data. Both the "pre-training and finetuning" and the analysis of the response by the LLM can be run through pipeline.py.

The list of properties/descriptors is stored in /Pipeline/LLMqueries/txt_files/descriptors.txt. NOTE: some of the RDKit calculators are deprecated, thus resulting in some of the properties being non-computable. These deprecated calculators are handled in RDKitDescriptors.py.

Properties are computed through either RDKit, Modred, or a combination of autodE and ORCA.
Molecules are by default represented as SMILES strings. RDKit and Mordred calculations use their own format, Mol, as input. For properties that are computed through either RDKit or Mordred, the list of molecules to compute from are first converted from SMILES to Mol by RDKit's built-in method Chem.MolFromSmiles(SMILES_STR). The molecules are kept as SMILES strings for properties that are computed through autodE and ORCA.