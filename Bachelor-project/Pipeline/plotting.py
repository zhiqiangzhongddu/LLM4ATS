import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import argparse as ap
from descriptor_calculators.MordredDescriptors import OurMordredClass
from descriptor_calculators.RDKitDescriptors import get_RDKit_values
from rdkit import Chem
import pathlib
import string


path_to_dir = pathlib.Path(__file__).parent.resolve()

# (OLD) only works for tox21
def get_idx_of_target(target):
    if target == "NR-AR": return 0
    elif target == "NR-AR-LBD": return 1
    elif target == "NR-AhR": return 2
    elif target == "NR-Aromatase": return 3
    elif target == "NR-ER": return 4
    elif target == "NR-ER-LBD": return 5
    elif target == "NR-PPAR-gamma": return 6
    elif target == "SR-ARE": return 7
    elif target == "SR-ATADS": return 8
    elif target == "SR-HSE": return 9
    elif target == "SR-MMP": return 10
    elif target == "SR-p53": return 11

# (OLD) only works for tox21
def get_name_of_target(target):
    if target == 0: return "NR-AR"
    elif target == 1: return "NR-AR-LBD"
    elif target == 2: return "NR-AhR"
    elif target == 3: return "NR-Aromatase"
    elif target == 4: return "NR-ER"
    elif target == 5: return "NR-ER-LBD"
    elif target == 6: return "NR-PPAR-gamma"
    elif target == 7: return "SR-ARE"
    elif target == 8: return "SR-ATADS"
    elif target == 9: return "SR-HSE"
    elif target == 10: return "SR-MMP"
    elif target == 11: return "SR-p53"

def is_value_valid(val):
    if type(val) == float or type(val) == int or type(val) == np.float64 or type(val) == bool: return True
    else: return False

def extract_data(target_task, idx=0):
    labels = []
    smiles = []
    with open(str(path_to_dir) + f'/../Pipeline/data/ogbg_mol{target_task}/mapping/{target_task}.csv', mode='r') as file:
        chemfile = csv.reader(file)
        for nr, line in enumerate(chemfile):
            if nr == 0 or not line[idx]:
                continue
            labels.append(line[idx])
            smiles.append(line[-2])
    return labels, smiles

# (OLD)
def get_random_subset(labels, smiles, size, static_seed):
    seed = 42
    if not static_seed:
        seed = random.randint(1, 1000)
    random.seed(seed)
    sub_lab = random.sample(labels, size)
    random.seed(seed)
    sub_smi = random.sample(smiles, size)
    return sub_lab, sub_smi

def calc_avg(values, labels):
    avg0, avg1 = 0, 0
    count0, count1 = 0, 0
    for i, v in enumerate(values):
        if not is_value_valid(v): print(v); continue
        if bool(int(labels[i])): avg1 += v; count1 += 1
        else: avg0 += v; count0 += 1
    return avg0/count0, avg1/count1

def shorten_prop_name(p, all_props):
    words = p.split()
    max_words = 3
    
    if len(words) > max_words:
        p = ""
        for w in words[:max_words]:
            p = p + w + " "
        p = p[:-1] + "..."
    return p


# labels and smiles: list of 12 lists, each 'number of labeled mols for that target' long.
# labels: the labels of the labeled molecules.
# smiles: the smiles strings of the labeled molecules.
def tox21_plot(labels, smiles, aux_tasks, all_mord_descriptors, form, random_props, seed):
    print("Performing analysis of the properties suggested by the LLM")
    fig, ax = plt.subplots()
    mols = []
    # Compute mol format from each smiles string
    for i, t_smiles in enumerate(smiles):
        mols.append([])
        for s in t_smiles:
            mols[i].append(Chem.MolFromSmiles(s))
    results = []
    targets = []
    total_tasks = aux_tasks[0] + aux_tasks[1]
    f = open(str(path_to_dir) + "/response_analysis/tox21/results.txt", "w+")
    all_results_str = ""
    for i, t_labels in enumerate(labels):
        # v has size 'number of aux tasks' x 'number of mols'
        MordClass = OurMordredClass(aux_tasks[0], all_mord_descriptors)
        mord_vals = MordClass.get_mordred_values(mols[i])
        rdkit_vals = get_RDKit_values(mols[i], aux_tasks[1])

        v1 = np.transpose(np.array(mord_vals))
        print(v1.shape)
        v2= np.transpose(np.array(rdkit_vals))
        print(v2.shape)
        v = np.concatenate((v1, v2), axis=0)
        print(v.shape)
        t_name = get_name_of_target(i)
        targets.append(t_name)
        bio_target_str = f"\nBiological target {t_name} has {len(v[0])} labeled molecules"
        print(bio_target_str)
        diffs = []
        results_str = ""
        for j, p in enumerate(total_tasks):
            #values = v[j]/np.amax(v[j])
            values = v[j]
            # Compute the average value of props of toxic mols vs non-toxic mols
            avg0, avg1 = calc_avg(values, t_labels)
            diff = (avg1/avg0)*100
            diffs.append(diff)
            res_str = f"Average value of '{p}' for non-toxic mols: {avg0}\nAverage value of '{p}' for toxic mols: {avg1}\nDifference: {diff:.2f}%"
            results_str += res_str + "\n\n"
            print(results_str)
        all_results_str += bio_target_str + "\n\n" + results_str + "\n\n----------------------------------------\n"
        results.append(diffs)
    f.write(all_results_str)
    f.close()
    # Transpose the result such that each row corresponds to an auxiliary task
    res = np.transpose(np.array(results))
    print("Plotting result of analysis")
    ax.grid(True)
    for i, p in enumerate(res):
        aux_name = shorten_prop_name(total_tasks[i], total_tasks)
        ax.scatter(x=targets, y=np.log(p), s=30, label=aux_name)
    ax.legend()
    plt.title("Relative difference in computed value of labeled data from tox21"); plt.xlabel("Biological targets"); plt.ylabel("The relative difference of aux. props.")
    ticks = [3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4]
    plt.xticks(rotation=45, ha='right'); plt.yticks(ticks=np.array(ticks), labels=[f"{np.exp(i):.2f}%" for i in ticks])
    plt.axhline(y = np.log(100), color = 'r', linestyle = '-') 
    if random_props: plt.savefig(str(path_to_dir) + f"/response_analysis/tox21/seed_{seed}.png", bbox_inches='tight')
    else: plt.savefig(str(path_to_dir) + f"/response_analysis/tox21/{form}_tasks_{len(total_tasks)}.png", bbox_inches='tight')
    plt.show()


def get_task_description(target):
    if target == "tox21": return "level of toxicity"
    if target == "bbbp": return "blood-brain barrier permeability"
    if target == "esol": return "solubility"
    if target == "lipo": return "lipophilicity"
    else: raise ValueError("Invalid target task")



def one_label_reg_plot(labels, smiles, aux_tasks, all_mord_descriptors, target_task, form, random_props, seed):
    labels = [float(i) for i in labels]
    total_tasks = aux_tasks[0] + aux_tasks[1]

    fig, axs = plt.subplots(len(total_tasks), sharex=True)
    mols = []
    # Compute mol format from each smiles string
    for i in smiles:
        mols.append(Chem.MolFromSmiles(i))
    
    # mord_vals and rdkit_vals has size 'number of mols' x 'number of aux tasks'
    MordClass = OurMordredClass(aux_tasks[0], all_mord_descriptors)
    mord_vals = MordClass.get_mordred_values(mols)
    rdkit_vals = get_RDKit_values(mols, aux_tasks[1])

    v1 = np.transpose(np.array(mord_vals))
    print(v1.shape)
    v2= np.transpose(np.array(rdkit_vals))
    print(v2.shape)
    v = np.concatenate((v1, v2), axis=0)
    print(v.shape)
    colors = ["b","g","r","c","m","y","k"]
    slopes = []
    v_wo_nan_vals = []
    l_wo_nan_vals = []
    for j, p in enumerate(total_tasks):
        values = v[j]
        # Skip if all values is an error msg
        if not is_value_valid(values[0]) and not is_value_valid(values[1]) and not is_value_valid(values[2]): continue
        vals_wo_nans = []
        labs_wo_nan_vals = []
        for i, val in enumerate(values):
            if is_value_valid(val):vals_wo_nans.append(val); labs_wo_nan_vals.append(labels[i])
        vals_wo_nans = np.array(vals_wo_nans); v_wo_nan_vals.append(vals_wo_nans)
        labs_wo_nan_vals = np.array(labs_wo_nan_vals); l_wo_nan_vals.append(labs_wo_nan_vals)
        m, b = np.polyfit(labs_wo_nan_vals/np.amax(labs_wo_nan_vals), (vals_wo_nans/np.amax(vals_wo_nans)).astype('float'), 1)     
        slopes.append(m)    
        m, b = np.polyfit(labs_wo_nan_vals, vals_wo_nans.astype('float'), 1)
        p = shorten_prop_name(p, total_tasks)
        axs[j].scatter(x=labels, y=values, s=2, label=p, color=colors[j%len(colors)])
        axs[j].axline(xy1=(0, b), slope=m, color='k')
        axs[j].legend()
    
    # Plot the value of the pre_training task as a function of the molecule label
    fig.suptitle(f"Auxiliary task value as a function of {get_task_description(target_task)} of molecules."); plt.xlabel(f"{get_task_description(target_task)} of molecules")
    amax = np.amax(np.array(labels))
    amin = np.amin(np.array(labels))
    get_mid = lambda amax,amin: amin+(amax-amin)/2
    mid = get_mid(amax, amin)
    deci = 2
    ticks = [round(amin,deci), round(get_mid(mid,amin),deci), round(mid,deci), round(get_mid(amax,mid),deci), round(amax,deci)]
    plt.xticks(ticks=np.array(ticks), labels=np.array(ticks))
    if random_props:
        plt.savefig(str(path_to_dir) + f"/response_analysis/{target_task}/seed_{seed}.png", bbox_inches='tight')
        f = open(str(path_to_dir) + f"/response_analysis/{target_task}/seed_{seed}.txt", "w+")
    else:
        plt.savefig(str(path_to_dir) + f"/response_analysis/{target_task}/{form}_tasks_{len(total_tasks)}.png", bbox_inches='tight')
        f = open(str(path_to_dir) + f"/response_analysis/{target_task}/results.txt", "w+")
    # Create a string containing the results and write to file
    if len(total_tasks) != len(slopes): f.write("One or more auxiliary properties cannot be computed")
    else:
        res_str = ""
        for i, p in enumerate(total_tasks):
            # Compute the correlation coefficients
            correlation_coefficients = calc_correlation(l_wo_nan_vals[i], [v_wo_nan_vals[i]])
            res_str += f"{p}\nSlope: {slopes[i]:.4f}\nCorrelation coefficient: {correlation_coefficients[0]:.4f}\n\n"
        f.write(res_str)
    f.close()
    plt.show()


def one_label_clas_plot(labels, smiles, aux_tasks, all_mord_descriptors, target_task, form, random_props, seed):
    fig, ax = plt.subplots()
    mols = []
    # Compute mol format from each smiles string
    for i in smiles:
        mols.append(Chem.MolFromSmiles(i))
    
    total_tasks = aux_tasks[0] + aux_tasks[1]

    MordClass = OurMordredClass(aux_tasks[0], all_mord_descriptors)
    mord_vals = MordClass.get_mordred_values(mols)
    rdkit_vals = get_RDKit_values(mols, aux_tasks[1])

    v1 = np.transpose(np.array(mord_vals))
    print(v1.shape)
    v2= np.transpose(np.array(rdkit_vals))
    print(v2.shape)
    v = np.concatenate((v1, v2), axis=0)
    print(v.shape)
    results = []
    for j, p in enumerate(total_tasks):
        values = v[j]
        avg0, avg1 = calc_avg(values, labels)
        diff = (avg1/avg0)*100
        results.append(diff)
    ax.grid(True)
    alphabet = [(f"prop {p}") for p in string.ascii_lowercase]
    ax.scatter(x=alphabet[:len(total_tasks)], y=np.log(results))
    #ax.scatter(x=total_tasks, y=np.log(results))
    plt.title(f"Relative difference in computed value of labeled data from {target_task}"); plt.xlabel("Auxiliary targets"); plt.ylabel("The relative difference of aux. props.")
    ticks = [3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2]
    plt.xticks(rotation=45, ha='right'); plt.yticks(ticks=np.array(ticks), labels=[f"{np.exp(i):.2f}%" for i in ticks])
    plt.axhline(y = np.log(100), color = 'r', linestyle = '-') 
    if random_props: plt.savefig(str(path_to_dir) + f"/response_analysis/{target_task}/seed_{seed}.png", bbox_inches='tight')
    else: plt.savefig(str(path_to_dir) + f"/response_analysis/{target_task}/{form}_tasks_{len(total_tasks)}.png", bbox_inches='tight')
    plt.show()

# Function to be called from within pipeline.py
def analyse(aux_tasks, all_mord_descriptors, target_task, form, random_props, seed):
    labels, smiles = [], []

    if target_task == "tox21":
        for bio_target in range(12): # 12 biological targets
            l, s = extract_data(target_task, bio_target)
            l = np.array([int(float(i)) for i in l])
            labels.append(l)
            smiles.append(s)
        tox21_plot(labels, smiles, aux_tasks, all_mord_descriptors, form, random_props, seed)

    if target_task == "bbbp":
        l, s = extract_data(target_task)
        one_label_clas_plot(l, s, aux_tasks, all_mord_descriptors, target_task, form, random_props, seed)

    if target_task == "esol" or target_task == "lipo":
        l, s = extract_data(target_task)
        ccat = np.concatenate((np.transpose(np.array(l)[:, np.newaxis]), np.transpose(np.array(s)[:, np.newaxis])), axis=0) 
        labels_and_smiles = np.transpose(np.array(sorted(ccat.T, key=lambda x: x[0])))
        one_label_reg_plot(labels_and_smiles[0], labels_and_smiles[1], aux_tasks, all_mord_descriptors, target_task, form, random_props, seed)

# Input: Some list of numbers
# Returns: Sum(list)/len(list)
def calc_sample_avg(list):
    sum = 0
    for num in list:
        sum += num
    return sum/len(list)

# Input: Two lists of number
def calc_covariance(x, y):
    # Calculate sample averages
    x_avg = calc_sample_avg(x)
    y_avg = calc_sample_avg(y)

    # Calculate the difference between sample avg and the actual value
    def helper(list, avg):
        out = []
        for i in list: 
            out.append((i - avg))
        return out
    
    x_diff = helper(x, x_avg)
    y_diff = helper(y, y_avg)

    # Multiply the elements of each list together
    temp = []
    for i in range(len(x_diff)):
        temp.append(x_diff[i] * y_diff[i])
    
    # Take the average of the resulting list to obtain the covariance
    # This is all inspired by probabilitycourse.com, where the definition of covariance is described
    return calc_sample_avg(temp)


# Input: Expects a list of the target labels and a matrix of all the aux_target values.
# Matrix should be sorted in the same manner that we sort the list of aux_target names.
# Returns a list of correlation co-efficients between the aux_task value and the target task value.
def calc_correlation(labels, aux_tasks):

    # By definition, covariance(x, x) = Var(x)
    label_variance = calc_covariance(labels, labels)

    # List definitions
    aux_variances = []
    aux_correlation_coefficients = []

    # Calculation loop
    for i in range(len(aux_tasks)):
        # By definition, covariance(x, x) = Var(x)
        aux_variances.append(calc_covariance(aux_tasks[i], aux_tasks[i]))

        # By definition, sqrt(Var(x)) = sigma(x)
        aux_correlation_coefficients.append(
            calc_covariance(labels, aux_tasks[i])/((label_variance*aux_variances[i])**0.5)
        )
    return aux_correlation_coefficients

