# Large Language Model for Auxiliary Task Selection

## Preparation

### Benchmark Datasets

1. Molecular Datasets
   - We adopt datasets from [OGB Graph Property Prediction Benchmark](https://ogb.stanford.edu/docs/graphprop/), including ogbg-mollipo, ogbg-molesol, ogbg-moltox21 and ogbg-bbbp.
   - The public splits and evaluation metrics are integrated in. 
   - Each molecule is represented using a SMILES string (available at `data/ogbg_molbace/mapping/mol.csv.gz`)
2. More?


## Instruction to execution

1. Auxiliary Task Collection
   1. Molecular Descriptor
      1. RDKit Computation Property. 
         Property explanations are available at `code/descriptors/rdkit_property`: rdkit_property_book. 
         Load the file and print it. 
      2. Physical Computation Property. 
         Property explanations are available at `code/descriptors/physical_property`: _physical_molecular_property_book
   2. More?
2. Query LLMs for Auxiliary Task Selection
   1. TBD. 
3. Auxiliary Task-enhanced Property Prediction
   1. Scratch
   2. Train
   3. Pre-train


## Example execution code

### Check Computation Property
Example: 
```
# Check all computation properties
python -m code.descriptors.property
# Check RDKit computation properties
python -m code.descriptors.rdkit_property
# Check RDKit computation properties
python -m code.descriptors.physical_property
```

### Query LLMs for Auxiliary Task Selection
Example: 
```
```

### Auxiliary Task-enhanced Property Prediction
Example: 
```
python -m code.main_GNN dataset ogbg-molbace task.mode scratch gnn.model.name gin-v seed 42
```