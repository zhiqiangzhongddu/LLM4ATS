from rdkit import Chem
import pandas as pd
import time
from MordredDescriptors import OurMordredClass
from LLMClass import our_LLM_class
from RDKitDescriptors import get_RDKit_values
import argparse as ap
import pathlib
import numpy as np
from QMproperties.QMproperties import get_QMprops_from_list
path_to_dir = pathlib.Path(__file__).parent.resolve()


# Update the current table of computed values
def add_vals_to_table(table_of_vals, values):
    for i in range(len(values)):
        table_of_vals[i] += values[i]
    return table_of_vals



if __name__ == "__main__":
    parser = ap.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--print_props_to_file', type=str, default="", help=f'filename to output the selected properties, type=str')
    parser.add_argument('--print_values_to_file', type=str, default='', help =f'filename to output the computed values, type=str')
    parser.add_argument('--aux_tasks', type=int, default=5, help=f'number of auxiliary tasks, type=bool')
    parser.add_argument('--Mordred', type=bool, default=True, help=f'whether to include Mordred descriptors, type=bool')
    parser.add_argument('--RDKit', type=bool, default=True, help=f'whether to include RDKit descriptors, type=bool')
    parser.add_argument('--QM', type=bool, default=True, help='whether to include QM properties, type=bool')
    parser.add_argument('--MoleculeDataset', type=str, default="moleculesTestDS.txt", help='filename of the molecule file, type=str')
    parser.add_argument('--target_task', type=str, default="level of toxicity", help='name of the target task, type=str')
    parser.add_argument('--Train', type=bool, default=False, help='whether to train the model, type=bool')
    parser.add_argument('--Test', type=bool, default=False, help='whether to test the model, type=bool')
    parser.add_argument('--random_properties', type=bool, default=True, help="Whether the properties are randomly selected, type=bool")
    parser.add_argument('--save_conversation', type=bool, default=False, help="Whether to save the conversation, type=bool")
    parser.add_argument('--conversation_file', type=str, default='conversation.txt', help="Name of the conversation file, type=str")
    parser.add_argument('--print_time', type=bool, default=True, help="Whether to print the time, type=bool")
    args = parser.parse_args()
       
    LLMClass = our_LLM_class()
    LLMClass.aux_tasks = args.aux_tasks
    LLMClass.target_task = args.target_task
    LLMClass.all_desc_inc['Mordred 2D descriptors'] = args.Mordred
    LLMClass.all_desc_inc['RDKit 2D descriptors'] = args.RDKit
    LLMClass.all_desc_inc['QM properties'] = args.QM
    LLMClass.save_conversation = args.save_conversation
    LLMClass.conversation_file = args.conversation_file

    # Select some random properties if arg is set to True, else ask the LLM for suggestions
    if args.random_properties:
        props = LLMClass.get_random_properties()
    else:
        props = LLMClass.run_properties_tournament()
    
    # Add all the selected properties to a dict, segregated by source
    table_of_props = {}
    for prop in props:
        for key in LLMClass.all_descriptors:
            if prop in LLMClass.all_descriptors[key]:
                if key not in table_of_props:
                    table_of_props[key] = []
                table_of_props[key].append(prop)
                break
    
    # Prints the selected properties to a given file if arg is set
    if args.print_props_to_file != '':
        print("Printing properties to file")
        f = open(str(path_to_dir) + "/" +args.print_props_to_file, 'w')
        res = ''
        for i in props: res+=i+'\n'
        f.write(res[:-2])
        f.close()
        
    
    # Read the molecules from the file and convert them to RDKit molecules, mols
    f = open(str(path_to_dir) + "/../Data/" + args.MoleculeDataset, "r")
    smiles = f.read().split("\n")
    mols = []
    for i in smiles:
        mols.append(Chem.MolFromSmiles(i))
    
    # Create an empty table of computed values
    table_of_vals = [[] for i in range(len(mols))]

    
    # Calculate the values of the mordred descriptors
    if "Mordred 2D descriptors" in table_of_props.keys():
        MordClass = OurMordredClass(table_of_props["Modred 2D descriptors"])
        mordvals = MordClass.get_mordred_values(mols)
        table_of_vals = add_vals_to_table(table_of_vals, mordvals)
    
    # Calculate the values of the RDKit descriptors
    if "RDKit 2D descriptors" in table_of_props.keys():
        rdkitvalues = get_RDKit_values(mols, table_of_props["RDKit 2D descriptors"])
        table_of_vals = add_vals_to_table(table_of_vals, rdkitvalues)
    
    # Calculate the values of the QM properties
    if "QM properties" in table_of_props.keys():
        QMValues = get_QMprops_from_list(smiles, table_of_props["QM properties"])
        table_of_vals = add_vals_to_table(table_of_vals, QMValues)

    # THIS IS WHERE YOU CAN IMPLEMENT YOUR OWN PIPELINE FOR CALCULATING DESCRIPTORS
    
    # THIS IS WHERE YOU CAN IMPLEMENT YOUR OWN PIPELINE FOR CALCULATING DESCRIPTORS

    # Prints the computed values to a given file if arg is set
    if args.print_values_to_file != '':
        print("Printing values to file")
        f = open(str(path_to_dir) +"/" +args.print_values_to_file, 'w')
        res_lst = []
        res = ""
        for i_index,i in enumerate(table_of_vals):
            res_lst.append([])
            for j in range(len(i)):
                # Replaces error messages with NA for values not computable for a given molecule
                if type(i[j]) == float or type(i[j]) == int or type(i[j]) == np.float64:
                    res += str(i[j]) + ","
                    res_lst[i_index].append(i[j])
                else:
                    res += "NA,"
                    res_lst[i_index].append(np.nan)
            res = res[:-1] + "\n"
        f.write(res)
        f.close()
    
    
    if args.Train:
        # implement code section for training
        pass
    
    if args.Test:
        # Implement code section for testing
        pass
    
    
    
    

    
    
    


