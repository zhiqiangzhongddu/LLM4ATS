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

By default, the properties are selected by the LLM. If '--random_properties' is set, the properties will be selected randomly. NOTE: It is possible to use previously selected properties by setting the '--use_prev_llm_props' flag. Otherwise the LLM will be asked to select new properties. The results of previous queries and their settings are stored in /Pipeline/LLMqueries/txt_files/counters. When '--use_prev_llm_props' is set, a list of 'aux_tasks' properties will be returned if there is a previous query response matching the current settings (target_task, msg_form, NOTour, max_props). If there is no previous response from the LLM using these settings, the returned list will be empty. The conversation with chat-GPT can be written to a file with '--save_conversation'.

Currently, the datasets are stored in /Pipeline/data. Both the "pre-training and finetuning" and the analysis of the response by the LLM can be run through pipeline.py. The properties to be computed are either the properties selected by the LLM right after it has been queried, previously selected properties if '--use_prev_llm_props' is set, or randomly selected properties if '--random_properties' is set. The properties are computed for the molecules in the dataset specified by the '--target_task' setting. To obtain the same computed values for the same properties as we have used in pre-training, simply give the '--use_prev_llm_props' flag and specify the taget task/dataset to be used.

Pre-training and normal training can be done by setting the corresponding flags. When pre-training, the properties to be used are the above-mentioned selected properties.

The list of properties/descriptors is stored in /Pipeline/LLMqueries/txt_files/descriptors.txt. NOTE: some of the RDKit calculators are deprecated, thus resulting in some of the properties being non-computable. These deprecated calculators are handled in RDKitDescriptors.py.

Properties are computed through either RDKit, Modred, or a combination of autodE and ORCA.
Molecules are by default represented as SMILES strings. RDKit and Mordred calculations use their own format, Mol, as input. For properties that are computed through either RDKit or Mordred, the list of molecules to compute from are first converted from SMILES to Mol by RDKit's built-in method Chem.MolFromSmiles(SMILES_STR). The molecules are kept as SMILES strings for properties that are computed through autodE and ORCA.