# Large Language Model Agents for Auxiliary Task Selection

## Project Description

The performance of Machine Learning (ML) models is often limited by the availability of labeled data. 
In many real-world scenarios, labeling data is both expensive and time-consuming, particularly 
in fields such as biology and chemistry. 
To address this, some recent studies suggest using computable properties as auxiliary tasks to 
enhance ML models during pre-training and training. 
However, selecting these auxiliary tasks often requires domain expertise, which may not always 
be readily available.

This project aims to create an automated pipeline that leverages Large Language Models (LLMs) 
to select auxiliary tasks effectively. 
The proposed project consists of the following steps:
1. **Auxiliary Task Collection**: 
   Gather a comprehensive set of available auxiliary tasks.
2. **LLM-based Auxiliary Task Selection**:
   Use LLMs to identify the most suitable auxiliary tasks.
3. **Enhanced Property Prediction**: 
   Use selected auxiliary tasks to boost the prediction of key molecular properties. 

### Challenges:

1. **Unstable LLM-based Auxiliary Task Selection**:
   Our previous results indicate that LLMs often select auxiliary tasks in a seemingly random manner, 
   lacking consistency. 
   To mitigate this issue, integrating Retrieval-Augmented Generation (RAG) or 
   multi-agent frameworks may help achieve more reliable task selection.
2. **Learning from Noisy Signals**: 
   Even when LLM agents consistently identify auxiliary tasks, a significant amount of noise remains. 
   Specifically, assessing the correlation between the selected auxiliary tasks and the primary 
   target tasks is challenging. 
   Additionally, determining which information is useful and how to effectively use it adds further complexity. 
   To tackle this, we plan to employ Partial Label Learning techniques to better handle noisy labels 
   and extract valuable signals.


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