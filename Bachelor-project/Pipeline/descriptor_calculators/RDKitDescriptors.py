import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors,GraphDescriptors, Crippen, Lipinski, MolSurf, Fragments
from rdkit.Chem.EState import EState_VSA

def calc_RDKit_desc(mol, desc_name_list): 
    Lipinski.RingCount(mol)
    out = []
    graphDes = ["BalabanJ", "BertzCT", "Ipc", "Kappa 1", "Kappa 2", "Kappa 3", "Chi0n", "Chi1n", "Chi2n", "Chi3n","Chi4n", "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v"]
    CrippenDes = ["MolLogP", "MolMR"]
    EState_VSA_Des = ["EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "EState_VSA10", "EState_VSA11"]
    Descriptors_lst = ["HeavyAtomMolWt", "NOCount", "NumValenceElectrons", "ExactMolWT", "MolWT"]
    LipinskiDes = ["Heavy Atom Count", "NHOH group count", "NO group count", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",  "RingCount"]
    molSurfDes = ["TPSA", "LabuteASA", "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "SMR_VSA1", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SMR_VSA10", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12"]
    fragDes = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
    
    # List of deprecated calculators
    list_of_uncalc_desc = ["Topliss fragments", "FractionCSP3", "NumAmideBonds", "NumBridgeheadAtoms", "BCUT2D", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings", "MQNs", "Autocorr2D", "NumSpiroAtoms", "Phi"]
    
    for name in desc_name_list:
        name = name.split(":")[0]
        if name in graphDes: # or name in CrippenDes or name in LipinskiDes or name in molSurfDes:
            func = getattr(GraphDescriptors, name.replace(" ", ""))
            out.append(func(mol))
            continue
        if name in CrippenDes:
            func = getattr(Crippen, name.replace(" ", ""))
            out.append(func(mol))
            continue
        if name in LipinskiDes:
            func = getattr(Lipinski, name.replace(" ", "").replace("group", "").replace("count", "Count"))
            out.append(func(mol))
            continue
        if name in molSurfDes:
            if len(name)>4:
                if name[:len("PEOE_VSA")] == "PEOE_VSA": out.append(MolSurf.PEOE_VSA_(mol)[int(name[len("PEOE_VSA"):])-1]); continue
                if name[:len("SMR_VSA")] == "SMR_VSA": out.append(MolSurf.SMR_VSA_(mol)[int(name[len("SMR_VSA"):])-1]); continue
                if name[:len("SlogP_VSA")] == "SlogP_VSA": out.append(MolSurf.SlogP_VSA_(mol)[int(name[len("SlogP_VSA"):])-1]); continue
            func = getattr(MolSurf, name.replace(" ", ""))
            out.append(func(mol))
            continue
        if name in fragDes:
            func = getattr(Fragments, name)
            out.append(func(mol))
            continue
        if name in Descriptors_lst:
            func = getattr(Descriptors, name)
            out.append(func(mol))
            continue
        if name in EState_VSA_Des:
            if name[:len("EState_VSA")] == "EState_VSA":
                out.append(EState_VSA.EState_VSA_(mol)[int(name[len("EState_VSA"):])-1])
                continue
        if name in list_of_uncalc_desc:
            out.append("Not calculated")
            continue
        else:
            print("Descriptor not found: " + name) 
    
    return out

def get_RDKit_values(mol_list_in, desc_list_in):
    out = []
    for i in range(len(mol_list_in)):
        out += [calc_RDKit_desc(mol_list_in[i], desc_list_in)]
    return out
