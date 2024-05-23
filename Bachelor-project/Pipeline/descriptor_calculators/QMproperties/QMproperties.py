import subprocess
import autode as ade
ORCA = ade.methods.ORCA()
XTB = ade.methods.XTB()

def get_QMprops(SMI_str):
	tmp = ade.Molecule(name='molecule', smiles=SMI_str)
	kwds = ade.Config.ORCA.keywords
	ade.Config.ORCA.keywords.sp = ['B3LYP', 'def2-SVP']
	ade.Config.n_cores = 1
	kwds.opt_ts = ['%elprop\n' 'Polar 1\n' 'end']
	#tmp.optimise(method=XTB)
	#tmp.optimise(method=ORCA)
	#tmp.single_point(method=ORCA)
	tmp.calc_thermo(method=ORCA)
	#tmp.calc_thermo()
	ZPE = tmp.zpe
	G = tmp.free_energy
	H = tmp.enthalpy
	U = tmp.energy + ZPE
	print(f'U = {tmp.energy+tmp.zpe:.6f}', f'H = {tmp.enthalpy:.6f}', f'G = {tmp.free_energy:.6f}', f'ZPE = {tmp.zpe:.6f}',  f'units = {tmp.energy.units}')
	return [float(ZPE), float(U), float(H), float(G)]

def get_QMprops_from_list(SMI_lst, props):
	values = [get_QMprops(smi) for smi in SMI_lst]
	wanted_vals = []
	props_idx = []
	for p in props:
		if p == "Zero point vibration energy":
			props_idx.append(0)
		if p == "Internal energy at 298.15K":
			props_idx.append(1)
		if p == "Enthalpy at 298.15K":
			props_idx.append(2)
		if p == "Free energy at 298.15K":
			props_idx.append(3)
	for i, m in enumerate(values):
		wanted_vals.append([])
		for p in props_idx:
			wanted_vals[i].append(m[p])
	print(wanted_vals)
	subprocess.run(['ls'])
	#subprocess.run(['rm', '-v', '!(\"QMproperties.py\")'])
	return wanted_vals

#get_QMprops_from_list(['O'], ["Internal energy at 298.15K","Zero point vibration energy"])
