# Large Language Model for Auxiliary Task Selection

## Preparation

### Benchmark Datasets

1. Molecular Datasets
   - We adopt datasets from [OGB Graph Property Prediction Benchmark](https://ogb.stanford.edu/docs/graphprop/), including ogbg-mollipo, ogbg-molesol, ogbg-moltox21 and ogbg-bbbp.
   - The public splits and evaluation metrics are integrated in. 
2. More?


## Instruction to execution

1. Auxiliary Task Collection
   1. Molecular Auxiliary Tasks
   2. More?
2. Query LLMs for Auxiliary Task Selection
3. Auxiliary Task-enhanced Property Prediction
   1. Scratch
   2. Train
   3. Pre-train


## Example execution code

### Query LLMs for Auxiliary Task Selection
Example: 
```
```

### Auxiliary Task-enhanced Property Prediction
Example: 
```
python -m code.main_GNN dataset ogbg-molbace task.mode scratch seed 42
```