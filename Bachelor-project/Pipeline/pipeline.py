from rdkit import Chem
import pandas as pd
import time
import csv
from descriptor_calculators.MordredDescriptors import OurMordredClass
from LLMqueries.LLMClass import our_LLM_class
from LLMqueries.geminiLLMClass import our_gemini_LLM_class
from descriptor_calculators.RDKitDescriptors import get_RDKit_values
import argparse as ap
from plotting import analyse, get_task_description
from code.GNNs.gnn_pretrainer import GNNPreTrainer
from code.GNNs.gnn_trainer import GNNTrainer
from code.utils import set_seed
from code.config import cfg, update_cfg
import pathlib
import torch
import numpy as np
import random

# from code import main_PreTrainFineTune
# from descriptor_calculators.QMproperties.QMproperties import get_QMprops_from_list
path_to_dir = pathlib.Path(__file__).parent.resolve()


# Update the current table of computed values
def add_vals_to_table(table_of_vals, values):
    for i in range(len(values)):
        table_of_vals[i] += values[i] # concat the lists of lists
    return table_of_vals

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ap.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = ap.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--aux_tasks', type=int, default=5, help=f'Number of auxiliary tasks, type=int')
    parser.add_argument('--target_task', type=str, default="tox21", help='Abbreviated name of the target task. Possible tasks: tox21, bbbp, esol, lipo. type=str')
    parser.add_argument('--pretrain', type=bool, default=False, action=ap.BooleanOptionalAction, help='Whether to pretrain the model, type=bool')
    parser.add_argument('--train', type=bool, default=False, action=ap.BooleanOptionalAction, help='Whether to train the model, type=bool')
    parser.add_argument('--msg_form', type=str, default="form_4", help="Which message format to use when talking to the LLM, type=str")
    parser.add_argument('--print_time', type=bool, default=True, action=ap.BooleanOptionalAction, help="Whether to print the time, type=bool")
    parser.add_argument('--epochs', type=int, default=10, help="Specify the number of epochs to train over, type=int")
    parser.add_argument('--model', type=str, default="gin-v", help="Specify the model type to train, type=str")
    parser.add_argument('--LLM', type=str, default="openai", help="Specify the LLM to use, type=str, options=(\"openai\", \"gemini\")")

    # for testing
    parser.add_argument('--Mordred', type=bool, default=True, action=ap.BooleanOptionalAction, help=f'Whether to include Mordred descriptors, type=bool')
    parser.add_argument('--RDKit', type=bool, default=True, action=ap.BooleanOptionalAction, help=f'Whether to include RDKit descriptors, type=bool')
    parser.add_argument('--QM', type=bool, default=True, action=ap.BooleanOptionalAction, help='Whether to include QM properties, type=bool')
    parser.add_argument('--response_analysis', type=bool, default=False, action=ap.BooleanOptionalAction, help='Whether to run an analysis of the response of the LLM, type=bool')
    parser.add_argument('--print_props_to_file', type=str, default="", help=f'Filename to output the selected properties, type=str')
    parser.add_argument('--random_properties', type=bool, default=False, action=ap.BooleanOptionalAction, help="Whether the properties are randomly selected, type=bool")
    parser.add_argument('--save_conversation', type=bool, default=False, action=ap.BooleanOptionalAction, help="Whether to save the conversation with the LLM, type=bool")
    parser.add_argument('--conversation_file', type=str, default='conversation', help="Name of the conversation file, type=str")
    parser.add_argument('--use_prev_llm_props', type=bool, default=False, action=ap.BooleanOptionalAction, help="Whether to use the properties from the previous LLM attempts, type=bool")
    parser.add_argument('--NOTour' , type=int, default=10, help="# of tournaments, type=int")
    parser.add_argument('--rand_props_seed', type=int, default=random.randint(1, 10000), help="Seed to be used when selecting random properties, type=int")


    args = parser.parse_args()
    if args.LLM == "openai":
        LLMClass = our_LLM_class(path_to_dir)
    elif args.LLM == "gemini":
        LLMClass = our_gemini_LLM_class(path_to_dir)
    
    # LLMClass = our_LLM_class(path_to_dir)
    LLMClass.aux_tasks = args.aux_tasks
    LLMClass.set_message(args.msg_form)
    LLMClass.target_task = get_task_description(args.target_task)
    LLMClass.all_desc_inc['Mordred 2D descriptors'] = args.Mordred
    LLMClass.all_desc_inc['RDKit 2D descriptors'] = args.RDKit
    LLMClass.all_desc_inc['QM properties'] = args.QM
    LLMClass.save_conversation = args.save_conversation
    LLMClass.conversation_file = str(path_to_dir) +"/LLMqueries/txt_files/" +args.conversation_file+".txt"
    LLMClass.prev_prop_flag = args.use_prev_llm_props
    LLMClass.NO_tournaments = args.NOTour
    LLMClass.random_props_seed = args.rand_props_seed

    # Select some random properties if arg is set to True, else ask the LLM for suggestions
    if args.random_properties:
        props = LLMClass.get_random_properties()
    else:
        props = LLMClass.run_properties_tournament()


    # Add all the selected properties to a dict, segregated by source
    table_of_props = {}
    props = [i.lower() for i in props]

    #print(LLMClass.all_desc_wo_des)
    for prop in props:
        for key in LLMClass.all_descriptors:
            if prop in LLMClass.all_desc_wo_des[key]:
                if key not in table_of_props:
                    table_of_props[key] = []
                table_of_props[key].append(prop)
                break
    
    print("\nOut list content: ", *props, sep='\n')

    
    # Prints the selected properties to a given file if arg is set
    if args.print_props_to_file != '':
        print("Printing properties to file")
        f = open(str(path_to_dir) + "/LLMqueries/txt_files/" +args.print_props_to_file+".txt", 'w')
        res = ''
        for i in props: res+=i+'\n'
        f.write(res[:-2])
        f.close()
        
    
    # Read the molecules from the file and convert them to RDKit molecules, mols
    with open(str(path_to_dir) + f'/data/ogbg_mol{args.target_task}/mapping/{args.target_task}.csv', mode='r') as file:
        toxfile = csv.reader(file)
        smiles = [row[-2] for idx,row in enumerate(toxfile) if idx != 0] # select the SMILES string at the next-to-last position

    mols = []
    for i in smiles:
        mols.append(Chem.MolFromSmiles(i))
    
    # Create an empty table of computed values
    table_of_vals = [[] for _ in range(len(mols))]
    
    # Calculate the values of the mordred descriptors
    if "Mordred 2D descriptors" in table_of_props.keys():
        print("MORDRED")
        MordClass = OurMordredClass(table_of_props["Mordred 2D descriptors"],LLMClass.all_desc_wo_des["Mordred 2D descriptors"])
        mordvals = MordClass.get_mordred_values(mols)
        table_of_vals = add_vals_to_table(table_of_vals, mordvals)
    
    # Calculate the values of the RDKit descriptors
    if "RDKit 2D descriptors" in table_of_props.keys():
        print("RDKIT")
        # Return the lower case descriptor names to their original writing
        for j,e1 in enumerate(table_of_props["RDKit 2D descriptors"]):
            for i,e in enumerate(LLMClass.all_desc_wo_des["RDKit 2D descriptors"]):
                if e1 == e: table_of_props["RDKit 2D descriptors"][j] = LLMClass.all_descriptors["RDKit 2D descriptors"][i].split(":")[0]; break
        rdkitvalues = get_RDKit_values(mols, table_of_props["RDKit 2D descriptors"])
        table_of_vals = add_vals_to_table(table_of_vals, rdkitvalues)
    
    # Calculate the values of the QM properties
    # if "QM properties" in table_of_props.keys():
    #     QMValues = get_QMprops_from_list(smiles, table_of_props["QM properties"])
    #     table_of_vals = add_vals_to_table(table_of_vals, QMValues)

    # THIS IS WHERE YOU CAN IMPLEMENT YOUR OWN PIPELINE FOR CALCULATING DESCRIPTORS
    
    # THIS IS WHERE YOU CAN IMPLEMENT YOUR OWN PIPELINE FOR CALCULATING DESCRIPTORS    
    
    # Prints the computed values to a given file if arg is set
    # print(len(table_of_vals)*len(table_of_vals[0]))

        
    print("Printing values to file")
    f = open(str(path_to_dir) + f"/computed_values/{args.target_task}_vals.csv", 'w')
    res = ""
    props = []
    if "Mordred 2D descriptors" in table_of_props.keys(): props += table_of_props['Mordred 2D descriptors']
    if "RDKit 2D descriptors" in table_of_props.keys(): props += table_of_props['RDKit 2D descriptors']
    for prop in props: res += f"{prop},"
    res = res[:-1] + "\n"
    for i_index,e in enumerate(table_of_vals):
        for j in range(len(e)):
            # Replaces error messages with NA for values not computable for a given molecule
            if type(e[j]) == float or type(e[j]) == int or type(e[j]) == np.float64:
                res += str(e[j]) + ","
            elif type(e[j]) == bool:
                res += str(int(e[j])) + ","
            else:
                res += "NA,"
        res = res[:-1] + "\n"
    f.write(res)
    f.close()
        

    if args.response_analysis:
        list_of_props = []
        if "Mordred 2D descriptors" in table_of_props.keys(): list_of_props.append(table_of_props['Mordred 2D descriptors'])
        else: list_of_props.append([])
        if "RDKit 2D descriptors" in table_of_props.keys(): list_of_props.append(table_of_props['RDKit 2D descriptors'])
        else: list_of_props.append([])
        analyse(list_of_props, LLMClass.all_desc_wo_des["Mordred 2D descriptors"], args.target_task, args.msg_form, args.random_properties, args.rand_props_seed)



    if args.pretrain:
        set_seed(cfg.seed)
        cfg.dataset = f"ogbg-mol{args.target_task}"
        cfg.gnn.train.epochs = args.epochs
        cfg.gnn.model.name = args.model
        # Extract the computed values
        df = pd.read_csv(str(path_to_dir) + f'/computed_values/{args.target_task}_vals.csv')
        print("Properties usable: ",df.shape[1])

        # print("df.count: ", df.count())
        df = df.dropna(axis=1, how='all')
        # Create a tensor storing the table of values to pass to the GNN pretrainer
        pretrain_values = torch.tensor(df.values)
        # Run the pre-training and finetuning of the model
        gnn_pretrainer = GNNPreTrainer(cfg=cfg, aux_values=pretrain_values, rand_props_seed = args.rand_props_seed, use_QM9=False, one_at_a_time=False, random_props=args.random_properties)
        gnn_pretrainer.pretrain_and_eval()
        gnn_pretrainer.finetune_and_eval()
    
    
    if args.train:
        set_seed(cfg.seed)
        cfg.dataset = f"ogbg-mol{args.target_task}"
        cfg.gnn.train.epochs = args.epochs
        cfg.gnn.model.name = args.model
        gnn_trainer = GNNTrainer(cfg=cfg)
        gnn_trainer.train_and_eval()
    
