from __future__ import print_function, division
from pkg_resources import resource_filename
import os
import requests
import json
import torch
import pickle
import warnings
import functools
import numpy as np
import torch.nn as nn
from ase.io import read
from pymatgen.io.cif import CifWriter
import pymatgen.core as mg
from torch.autograd import Variable
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data import Dataset,DataLoader
from collections import defaultdict
from CifFile import ReadCif

warnings.filterwarnings("ignore")

package_directory = os.path.abspath(__file__).replace("pmcharge.py","")
files_to_download = {
                    'cm5.pth': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_cm5/cm5.pth',
                    'cm5.pkl': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_cm5/cm5.pkl',
                    'bader.pth': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_bader/bader.pth',
                    'bader.pkl': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_bader/bader.pkl',
                    'ddec.pth': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_ddec/ddec.pth',
                    'ddec.pkl': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_ddec/ddec.pkl',
                    'repeat.pth': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_repeat/repeat.pth',
                    'repeat.pkl': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_repeat/repeat.pkl',
                    'pbe.pth': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_pbe/pbe.pth',
                    'pbe.pkl': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_pbe/pbe.pkl',
                    'bandgap.pth': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_bandgap/bandgap.pth',
                    'bandgap.pkl': 'https://raw.githubusercontent.com/mtap-research/PACMAN-charge/main/pth/best_bandgap/bandgap.pkl'
                    }

for file_name, url in files_to_download.items():
    file_path = os.path.join(package_directory, file_name)
    if not os.path.exists(file_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_name} to {file_path}")
        else:
            print(f"Failed to download {file_name} from {url}")
    else:
        pass

import importlib
import sys
source = importlib.import_module('PACMANCharge')
sys.modules['model.utils'] = source
sys.modules['model'] = source
sys.modules['model4pre'] = source
sys.modules['source'] = source
sys.modules['source.utils'] = source

periodic_table_symbols = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
    'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def ase2pymatgen(cif_file):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        struc = read(cif_file)
        adaptor = AseAtomsAdaptor()
        pmg_structure = adaptor.get_structure(struc)
        cif_writer = CifWriter(pmg_structure)
        cif_writer.write_file(cif_file)

def CIF2json(cif_file):
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        structure = read(cif_file)
        struct = AseAtomsAdaptor.get_structure(structure)
    except:
        struct = CifParser(cif_file, occupancy_tolerance=10)
        struct.get_structures()
    _c_index, _n_index, _, n_distance = struct.get_neighbor_list(r=6, numerical_tol=0, exclude_self=True)
    _nonmax_idx = []
    for i in range(len(structure)):
        idx_i = (_c_index == i).nonzero()[0]
        idx_sorted = np.argsort(n_distance[idx_i])[: 200]
        _nonmax_idx.append(idx_i[idx_sorted])
    _nonmax_idx = np.concatenate(_nonmax_idx)
    index1 = _c_index[_nonmax_idx]
    index2 = _n_index[_nonmax_idx]
    dij = n_distance[_nonmax_idx]
    numbers = []
    try:
        elements = [str(site.specie) for site in struct.sites]
    except:
        elements = [str(site.species) for site in struct.sites]
    for i in range(len(elements)):
        ele = elements[i]
        atom_index = periodic_table_symbols.index(ele)
        numbers.append(int(int(atom_index)+1))
    nn_num = []
    for i in range(len(structure)):
        j = 0
        for idx in range(len(index1)):
            if index1[idx] == i:
                    j += 1
            else:
                    pass
        nn_num.append(j)
    data = {"rcut": 6.0,
            "numbers": numbers,
            "index1": index1.tolist(),
            "index2":index2.tolist(),
            "dij": dij.tolist(),
            "nn_num": nn_num}
    return data

def pre4pre(cif_file):
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        try:
            structure = mg.Structure.from_file(cif_file)
        except:
            try:
                atoms = read(cif_file)
                structure = AseAtomsAdaptor.get_structure(atoms)
            except:
                structure = CifParser(cif_file, occupancy_tolerance=10)
                structure.get_structures()
        coords = structure.frac_coords
        try:
            elements = [str(site.specie) for site in structure.sites]
        except:
            elements = [str(site.species) for site in structure.sites]
        pos = []
        for i in range(len(elements)):
            x = coords[i][0]
            y = coords[i][1]
            z = coords[i][2]
            pos.append([float(x),float(y),float(z)])
    except Exception as e:
        print(e)
    return pos

def average_and_replace(numbers, di):
    groups = defaultdict(list)
    for i, number in enumerate(numbers):
        if di ==3:
            key = format(number, '.3f')
            groups[key].append(i)
        elif di ==2:
            key = format(number, '.2f')
            groups[key].append(i)
    for key, indices in groups.items():
        avg = sum(numbers[i] for i in indices) / len(indices)
        for i in indices:
            numbers[i] = avg
    return numbers

def write4cif(cif_file,chg,digits,atom_type,neutral,charge_type,keep_connect):
    name = cif_file.split('.cif')[0]
    chg = chg.numpy()
    dia = int(digits)
    
    if atom_type:
        sum_chg = sum(chg)
        charges = []
        if neutral:
            charge = average_and_replace(chg,di=3)
            sum_chg = sum(charge)
            charges_1 = []
            for c in charge:
                cc = c - sum_chg/len(charge)
                charges_1.append(round(cc, dia))

            charge_2 = average_and_replace(charges_1,di=2)
            sum_chg = sum(charge_2)
            charges = []
            for c in charge_2:
                cc = c - sum_chg/len(charge_2)
                charges.append(round(cc, dia))
        else:
            charge = average_and_replace(chg,di=3)
            charges_1 = []
            for c in charge:
                charges_1.append(round(c, dia))
            charge_2 = average_and_replace(charges_1,di=2)
            charges = []
            for c in charge_2:
                charges.append(round(c, dia))

    else:
        sum_chg = sum(chg)
        charges = []
        if neutral:
            for c in chg:
                cc = c - sum_chg/len(chg)
                charges.append(round(cc, dia))
        else:
            for c in chg:
                charges.append(round(c, dia))
    if keep_connect:
       
        mof = ReadCif(name + ".cif")
        mof.first_block().AddToLoop("_atom_site_type_symbol",{'_atom_site_charge':[str(q) for q in charges]})
        with open(name + "_pacman.cif", 'w') as f:
            f.write("# " + charge_type + " charges by PACMAN v1.3.9 (https://github.com/mtap-research/PACMAN-charge/)\n" +
                    f"data_{name.split('/')[-1]}" + str(mof.first_block()))
        print("Compelete and save as "+ name + "_pacman.cif")

    else:
        with open(name + ".cif", 'r') as file:
            lines = file.readlines()
        lines[0] = "# " + charge_type + " charges by PACMAN v1.3.9 (https://github.com/mtap-research/PACMAN-charge/)\n"
        lines[1] = "data_"+name.split("/")[-1]+"_pacman\n"
        for i, line in enumerate(lines):
            if '_atom_site_occupancy' in line:
                lines.insert(i + 1, " _atom_site_charge\n")
                break
        charge_index = 0
        for j in range(i + 2, len(lines)):
            if charge_index < len(charges):
                lines[j] = lines[j].strip() + " " + str(charges[charge_index]) + "\n"
                charge_index += 1
            else:
                break
        with open(name + "_pacman.cif", 'w') as file:
            file.writelines(lines)
        file.close()
        with open(name + "_pacman.cif", 'r') as file:
            content = file.read()
        file.close()
        new_content = content.replace('_space_group_name_H-M_alt', '_symmetry_space_group_name_H-M')
        new_content = new_content.replace('_space_group_IT_number', '_symmetry_Int_Tables_number')
        new_content = new_content.replace('_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz')
        with open(name + "_pacman.cif", 'w') as file:
            file.write(new_content)
        file.close()
        print("Compelete and save as "+ name + "_pacman.cif")

def get_data_loader(dataset,batch_size,num_workers,collate_fn,pin_memory):
    data_loader = DataLoader(dataset,batch_size,shuffle=True,num_workers=num_workers,collate_fn=collate_fn,pin_memory=pin_memory)
    return data_loader

def collate_pool(dataset_list):
    batch_atom_fea = [] 
    batch_nbr_fea =[]
    batch_nbr_fea_idx1 = []
    batch_nbr_fea_idx2 = []
    batch_num_nbr = []
    crystal_atom_idx = []
    batch_pos = []
    batch_dij_ = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbr, dij_), (pos))\
        in enumerate(dataset_list):       
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea);batch_dij_.append(dij_)
        tt1 = np.array(nbr_fea_idx1)+base_idx
        tt2 = np.array(nbr_fea_idx2)+base_idx
        batch_nbr_fea_idx1.append(torch.LongTensor(tt1.tolist()))
        batch_nbr_fea_idx2.append(torch.LongTensor(tt2.tolist()))
        batch_num_nbr.append(num_nbr)
        crystal_atom_idx.append(torch.LongTensor([i]*n_i))
        batch_pos.append(pos)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx1, dim=0),torch.cat(batch_nbr_fea_idx2, dim=0),
            torch.cat(batch_num_nbr, dim=0),torch.cat(crystal_atom_idx,dim=0), torch.cat(batch_dij_,dim=0),
            torch.cat(batch_pos,dim=0))
            
class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var
    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)

class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}
    def get_atom_fea(self, atom_type):
        return self._embedding[atom_type]
    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
    def state_dict(self):
        return self._embedding
    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]
    
class AtomCustomJSONInitializer(AtomInitializer):
		def __init__(self, elem_embedding_file):
				elem_embedding = json.load(open(elem_embedding_file))
				elem_embedding = {int(key): value for key, value in elem_embedding.items()}
				atom_types = set(elem_embedding.keys())
				super(AtomCustomJSONInitializer, self).__init__(atom_types)
				for key in range(101):
						zz = np.zeros((101,))
						zz[key] = 1.0
						self._embedding[key] = zz.reshape(1,-1)
    
class CIFData(Dataset):
    def __init__(self,crystal_data,pos,radius,dmin,step):
        self.pos = pos
        self.radius = radius
        self.crystal_data = crystal_data
        atom_init_file = resource_filename('PACMANCharge', 'atom_init.json')
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
    def __len__(self):
        return 1
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self,_):
        nums = self.crystal_data['numbers']
        atom_fea = np.vstack([self.ari.get_atom_fea(nn) for nn in nums])
        pos = self.pos
        index1 = np.array(self.crystal_data['index1'])
        nbr_fea_idx = np.array(self.crystal_data['index2'])
        dij = np.array(self.crystal_data['dij']); dij_ = torch.Tensor(dij)
        nbr_fea = self.gdf.expand(dij)
        num_nbr = np.array(self.crystal_data['nn_num'])
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx1 = torch.LongTensor(index1)
        nbr_fea_idx2 = torch.LongTensor(nbr_fea_idx)
        num_nbr = torch.Tensor(num_nbr)
        pos = torch.Tensor(pos)
        return (atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbr,dij_), (pos)
    
class Normalizer(object):
	def __init__(self, tensor):
		self.mean = torch.mean(tensor)
		self.std = torch.std(tensor)
	def norm(self, tensor):
		return (tensor - self.mean) / self.std
	def denorm(self, normed_tensor):
		return normed_tensor * self.std + self.mean
	def state_dict(self):
		return {'mean': self.mean,'std': self.std}
	def load_state_dict(self, state_dict):
		self.mean = state_dict['mean']
		self.std = state_dict['std']
          
class ConvLayer(nn.Module):
    def __init__(self,atom_fea_len,nbr_fea_len):
        super(ConvLayer,self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.tanh_e = nn.Tanh()
        self.tanh_v = nn.Tanh()
        self.bn_v = nn.BatchNorm1d(self.atom_fea_len)
        self.phi_e = nn.Sequential(nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,self.atom_fea_len),
					nn.LeakyReLU(0.2),
					nn.Linear(self.atom_fea_len,self.atom_fea_len),
					nn.LeakyReLU(0.2),
					nn.Linear(self.atom_fea_len,self.atom_fea_len))
        self.phi_v = nn.Sequential(nn.Linear(2*self.atom_fea_len,self.atom_fea_len),
					nn.LeakyReLU(0.2),
					nn.Linear(self.atom_fea_len,self.atom_fea_len),
					nn.LeakyReLU(0.2),
					nn.Linear(self.atom_fea_len,self.atom_fea_len))
    def forward(self,atom_in_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx):
        N,M = atom_in_fea.shape
        atom_nbr_fea1 = atom_in_fea[nbr_fea_idx1,:]
        atom_nbr_fea2 = atom_in_fea[nbr_fea_idx2,:]
        nbr_num_fea = num_nbrs[nbr_fea_idx1].view(-1,1)
        total_nbr_fea = torch.cat([atom_nbr_fea1,atom_nbr_fea2,nbr_fea],dim=1)
        ek = self.phi_e(total_nbr_fea)
        rho_e_v = Variable(torch.zeros((N,M)).cuda() if torch.cuda.is_available() else torch.zeros((N,M)) ).scatter_add(0, nbr_fea_idx1.view(-1,1).repeat(1,M),ek/nbr_num_fea)
        total_node_fea = torch.cat([atom_in_fea,rho_e_v],dim=1)
        vi = self.phi_v(total_node_fea)		
        vi = self.bn_v(vi)
        ek = nbr_fea + ek
        vi = atom_in_fea + vi
        ek_sum = Variable(torch.zeros((N,M)).cuda() if torch.cuda.is_available() else torch.zeros((N,M))).scatter_add(0,nbr_fea_idx1.view(-1,1).repeat(1,M),ek/nbr_num_fea)
        Ncrys = torch.unique(crystal_atom_idx.view(-1,1)).shape[0]
        atom_nbr_fea = torch.cat([vi,ek_sum],dim=1) 
        global_fea = Variable(torch.zeros((Ncrys,2*M)).cuda() if torch.cuda.is_available() else torch.zeros((Ncrys,2*M)) ).scatter_add(0,crystal_atom_idx.view(-1,1).repeat(1,2*M),atom_nbr_fea)
        return ek,vi,global_fea,atom_nbr_fea

class SemiFullGN(nn.Module):
    def __init__(self,orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,n_feature):    
        super(SemiFullGN, self).__init__()
        self.node_embedding = nn.Linear(orig_atom_fea_len,atom_fea_len)
        self.edge_embedding = nn.Linear(nbr_fea_len,atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,nbr_fea_len=atom_fea_len) for _ in range(n_conv)])
        self.feature_embedding = nn.Sequential(nn.Linear(n_feature,512))
        self.atom_nbr_fea_embedding = nn.Sequential(nn.Linear(2*atom_fea_len,128))
        self.phi_pos = nn.Sequential(nn.Linear(512+128,512),
                                     nn.BatchNorm1d(512),
                                     nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(nn.Conv1d(64,512,3,stride=1,padding=0),nn.BatchNorm1d(512),nn.LeakyReLU(0.2),
                                   nn.Conv1d(512,512,3,stride=1,padding=0),nn.BatchNorm1d(512),nn.LeakyReLU(0.2),
                                   nn.Conv1d(512,256,3,stride=1,padding=1),nn.LeakyReLU(0.2),
                                   nn.Conv1d(256,256,3,stride=1,padding=1),nn.LeakyReLU(0.2),
                                   nn.Conv1d(256,1,kernel_size=4,stride=1,padding=0))
        self.cell_embedding = nn.Sequential(nn.Linear(9,128)) # remove
    def forward(self,atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,atom_idx,structure_feature):
        nbr_fea_idx1 = nbr_fea_idx1.cuda() if torch.cuda.is_available() else nbr_fea_idx1
        nbr_fea_idx2 = nbr_fea_idx2.cuda() if torch.cuda.is_available() else nbr_fea_idx2
        num_nbrs = num_nbrs.cuda() if torch.cuda.is_available() else num_nbrs
        atom_idx = atom_idx.cuda() if torch.cuda.is_available() else atom_idx
        atom_fea = atom_fea.cuda() if torch.cuda.is_available() else atom_fea
        nbr_fea = nbr_fea.cuda() if torch.cuda.is_available() else nbr_fea 
        structure_feature = structure_feature.cuda() if torch.cuda.is_available() else structure_feature
        atom_fea = self.node_embedding(atom_fea)
        nbr_fea = self.edge_embedding(nbr_fea)
        N,_ = atom_fea.shape 
        for conv_func in self.convs:
            nbr_fea,atom_fea,_,atom_nbr_fea = conv_func(atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,atom_idx)
        feature = structure_feature[atom_idx]
        feature = self.feature_embedding(feature)
        atom_nbr_fea = self.atom_nbr_fea_embedding(atom_nbr_fea)
        final_feature = torch.cat((atom_nbr_fea,feature),dim=-1)
        charge = self.phi_pos(final_feature)
        charge = charge.view(N,64,8)
        charge = self.conv(charge).squeeze()
        return charge
    
class ConvLayerE(nn.Module):
    def __init__(self,atom_fea_len,nbr_fea_len):
        super(ConvLayerE,self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.tanh_e = nn.Tanh()
        self.tanh_v = nn.Tanh()
        self.bn_v = nn.BatchNorm1d(self.atom_fea_len)
        self.phi_e = nn.Sequential(nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,self.atom_fea_len),
				    nn.LeakyReLU(0.2),
				    nn.Linear(self.atom_fea_len,self.atom_fea_len),
				    nn.LeakyReLU(0.2),
			        nn.Linear(self.atom_fea_len,self.atom_fea_len))
        self.phi_v = nn.Sequential(nn.Linear(2*self.atom_fea_len,self.atom_fea_len),
				    nn.LeakyReLU(0.2),
				    nn.Linear(self.atom_fea_len,self.atom_fea_len),
				    nn.LeakyReLU(0.2),
				    nn.Linear(self.atom_fea_len,self.atom_fea_len))
    def forward(self,atom_in_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx):
        N,M = atom_in_fea.shape
        atom_nbr_fea1 = atom_in_fea[nbr_fea_idx1,:]
        atom_nbr_fea2 = atom_in_fea[nbr_fea_idx2,:]
        nbr_num_fea = num_nbrs[nbr_fea_idx1].view(-1,1)
        total_nbr_fea = torch.cat([atom_nbr_fea1,atom_nbr_fea2,nbr_fea],dim=1)
        ek = self.phi_e(total_nbr_fea)
        rho_e_v = Variable(torch.zeros((N,M)).cuda() if torch.cuda.is_available() else torch.zeros((N,M))).scatter_add(0,nbr_fea_idx1.view(-1,1).repeat(1,M),ek/nbr_num_fea)
        total_node_fea = torch.cat([atom_in_fea,rho_e_v],dim=1)
        vi = self.phi_v(total_node_fea)		
        vi = self.bn_v(vi)
        ek = nbr_fea + ek
        vi = atom_in_fea + vi
        ek_sum = Variable(torch.zeros((N,M)).cuda() if torch.cuda.is_available() else torch.zeros((N,M)) ).scatter_add(0,nbr_fea_idx1.view(-1,1).repeat(1,M),ek/nbr_num_fea)
        Ncrys = torch.unique(crystal_atom_idx.view(-1,1)).shape[0]
        atom_nbr_fea = torch.cat([vi,ek_sum],dim=1)
        global_fea = Variable(torch.zeros((Ncrys,2*M)).cuda() if torch.cuda.is_available() else torch.zeros((Ncrys,2*M)) ).scatter_add(0,crystal_atom_idx.view(-1,1).repeat(1,2*M),atom_nbr_fea)
        return ek,vi,global_fea
    
class GCN(nn.Module):
    def __init__(self,orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,n_feature):    
        super(GCN, self).__init__()
        self.node_embedding = nn.Linear(orig_atom_fea_len,atom_fea_len).to(device)
        self.edge_embedding = nn.Linear(nbr_fea_len,atom_fea_len).to(device)
        self.convs = nn.ModuleList([ConvLayerE(atom_fea_len=atom_fea_len,nbr_fea_len=atom_fea_len).to(device) for _ in range(n_conv)])
        self.phi_u = nn.Sequential(nn.Linear(2*atom_fea_len,n_feature).to(device),nn.LeakyReLU(0.2).to(device),
				   nn.Linear(n_feature,n_feature).to(device),nn.Tanh().to(device))
    def Encoding(self,atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx):
        nbr_fea_idx1 = nbr_fea_idx1.cuda() if torch.cuda.is_available() else nbr_fea_idx1
        nbr_fea_idx2 = nbr_fea_idx2.cuda() if torch.cuda.is_available() else nbr_fea_idx2
        num_nbrs = num_nbrs.cuda() if torch.cuda.is_available() else num_nbrs
        crystal_atom_idx = crystal_atom_idx.cuda() if torch.cuda.is_available() else crystal_atom_idx
        atom_fea = atom_fea.cuda() if torch.cuda.is_available() else atom_fea
        nbr_fea = nbr_fea.cuda() if torch.cuda.is_available() else nbr_fea 
        atom_fea = self.node_embedding(atom_fea)
        nbr_fea = self.edge_embedding(nbr_fea)
        N,_ = atom_fea.shape
        Ncrys = torch.unique(crystal_atom_idx.view(-1,1)).shape[0]
        atom_nums_ = Variable(torch.ones((N,1)).cuda() if torch.cuda.is_available() else torch.ones((N,1)) )
        atom_nums = Variable(torch.zeros((Ncrys,1)).cuda() if torch.cuda.is_available() else torch.zeros((Ncrys,1)) ).scatter_add(0,crystal_atom_idx.view(-1,1),atom_nums_)
        N,_ = atom_fea.shape
        for conv_func in self.convs:
            nbr_fea,atom_fea,global_fea = conv_func(atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx)          
        global_fea = global_fea / atom_nums
        z = self.phi_u(global_fea)
        return z
    
class GCNE(nn.Module):
    def __init__(self,orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h):    
        super(GCNE, self).__init__()
        self.node_embedding = nn.Linear(orig_atom_fea_len,atom_fea_len)
        self.edge_embedding = nn.Linear(nbr_fea_len,atom_fea_len)
        self.convs = nn.ModuleList([ConvLayerE(atom_fea_len=atom_fea_len,nbr_fea_len=atom_fea_len) for _ in range(n_conv)])
        self.phi_u = nn.Sequential(nn.Linear(2*atom_fea_len,h_fea_len),nn.LeakyReLU(0.2),
				   nn.Linear(h_fea_len,h_fea_len),nn.Tanh())
        self.conv_to_fc = nn.Linear(h_fea_len, h_fea_len)
        self.conv_to_fc_lrelu = nn.LeakyReLU(0.2)
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
            self.activations = nn.ModuleList([nn.LeakyReLU(0.2) for _ in range(n_h-1)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(h_fea_len) for _ in range(n_h-1)])
        self.fc_out = nn.Linear(h_fea_len,1)
    def forward(self,atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx):
        z = self.Encoding(atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbrs, crystal_atom_idx)
        out = self.Regressor(z)
        return out
    def Encoding(self,atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx):
        nbr_fea_idx1 = nbr_fea_idx1.cuda() if torch.cuda.is_available() else nbr_fea_idx1
        nbr_fea_idx2 = nbr_fea_idx2.cuda() if torch.cuda.is_available() else nbr_fea_idx2
        num_nbrs = num_nbrs.cuda() if torch.cuda.is_available() else num_nbrs
        crystal_atom_idx = crystal_atom_idx.cuda() if torch.cuda.is_available() else crystal_atom_idx
        atom_fea = atom_fea.cuda() if torch.cuda.is_available() else atom_fea
        nbr_fea = nbr_fea.cuda() if torch.cuda.is_available() else nbr_fea 

        atom_fea = self.node_embedding(atom_fea)
        nbr_fea = self.edge_embedding(nbr_fea)
        N,_ = atom_fea.shape
        
        Ncrys = torch.unique(crystal_atom_idx.view(-1,1)).shape[0]
        atom_nums_ = Variable(torch.ones((N,1)).cuda() if torch.cuda.is_available() else torch.ones((N,1)) )
        atom_nums = Variable(torch.zeros((Ncrys,1)).cuda() if torch.cuda.is_available() else torch.zeros((Ncrys,1)) ).scatter_add(0,crystal_atom_idx.view(-1,1),atom_nums_)
        N,_ = atom_fea.shape
        for conv_func in self.convs:
            nbr_fea,atom_fea,global_fea = conv_func(atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx)          
        global_fea = global_fea / atom_nums
        z = self.phi_u(global_fea)
        return z
    def Regressor(self,z):
        crys_fea = self.conv_to_fc_lrelu(self.conv_to_fc(z))
        if hasattr(self,'fcs') and hasattr(self,'activations'):
            for fc,activation,_ in zip(self.fcs,self.activations,self.bns):
                crys_fea = activation(fc(crys_fea))
        out = self.fc_out(crys_fea)
        return out
    
def predict(cif_file,charge_type="DDEC6",digits=6,atom_type=True,neutral=True,keep_connect=True):
   
    if charge_type == "DDEC6":
        charge_model_name = package_directory + 'ddec.pth'
        nor_name = package_directory + 'ddec.pkl'
    elif charge_type == "Bader":
        charge_model_name = package_directory + 'bader.pth'
        nor_name = package_directory + 'bader.pkl'
    elif charge_type == "CM5":
        charge_model_name = package_directory + 'cm5.pth'
        nor_name = package_directory + 'cm5.pkl'
    elif charge_type == "REPEAT":
        charge_model_name = package_directory + 'repeat.pth'
        nor_name = package_directory + 'repeat.pkl'
    
    try:
        with open(nor_name, 'rb') as nor_f:
            charge_nor = pickle.load(nor_f)
        nor_f.close()
    except FileNotFoundError:
        print(f"Please use correct charge")

    print(f"CIF Name: {cif_file}")
    print(f"Charge Type: {charge_type}")
    print(f"Digits: {digits}")
    print(f"Atom Type: {atom_type}")
    print(f"Neutral: {neutral}")
    print(f"Keep Connect: {keep_connect}")

    try:
        if keep_connect:
            pass
        else:
            ase2pymatgen(cif_file)
        crystal_data = CIF2json(cif_file)
        pos=pre4pre(cif_file)
        dataset = CIFData(crystal_data,pos,6,0,0.2)
        loader= get_data_loader(dataset=dataset, batch_size=1, num_workers=0, collate_fn=collate_pool, pin_memory=False)
        for batch in loader:
            chg_1 = batch[0].shape[-1]+3
            chg_2 = batch[1].shape[-1]
            orig_atom_fea_len = batch[0].shape[-1]
        gcn = GCN(orig_atom_fea_len, chg_2, 128, 7, 256) 
        gcn.to(device)
        chkpt_ddec = torch.load(charge_model_name, map_location=torch.device(device))
        model4chg = SemiFullGN(chg_1,chg_2,128,8,256)
        model4chg.cuda() if torch.cuda.is_available() else model4chg.to(device)
        model4chg.load_state_dict(chkpt_ddec['state_dict'])
        model4chg.eval()
        for _, (input) in enumerate(loader):
            with torch.no_grad():
                input_var = (input[0].to(device),
                                input[1].to(device),
                                input[2].to(device),
                                input[3].to(device),
                                input[4].to(device),
                                input[5].to(device))
                encoder_feature = gcn.Encoding(*input_var)
                atoms_fea = torch.cat((input[0],input[7]),dim=-1)
                input_var2 = (atoms_fea.to(device),
                                input[1].to(device),
                                input[2].to(device),
                                input[3].to(device),
                                input[4].to(device),
                                input[5].to(device),
                                encoder_feature.to(device))
                chg = model4chg(*input_var2)
                chg = charge_nor.denorm(chg.data.cpu())
                write4cif(cif_file,chg,digits=digits,atom_type=atom_type,neutral=neutral,charge_type=charge_type,keep_connect=keep_connect)
    except Exception as e:
        print(e)

def n_atom(mof):
    structure = mg.Structure.from_file(mof)
    try:
        elements = [str(site.specie) for site in structure.sites]
    except:
         elements = [str(site.species) for site in structure.sites]
    return len(elements)

def Energy(cif_file):

    pbe_nor_name = package_directory+"pbe.pkl"
    pbe_model_name = package_directory+"pbe.pth"

    bandgap_nor_name = package_directory+"pbe.pkl"
    bandgap_model_name = package_directory+"pbe.pth"

    with open(pbe_nor_name, 'rb') as f:
        pbe_nor = pickle.load(f)
    f.close()
    with open(bandgap_nor_name, 'rb') as f:
        bandgap_nor = pickle.load(f)
    f.close()

    print(f"CIF Name: {cif_file}")

    try:
        struc = read(cif_file)
        crystal_data = CIF2json(struc,cif_file)
        pos=pre4pre(cif_file)
        num_atom = n_atom(cif_file)

        dataset = CIFData(crystal_data,pos,6,0,0.2)
        loader= get_data_loader(dataset=dataset, batch_size=1, num_workers=0, collate_fn=collate_pool, pin_memory=False)

        structures, _ = dataset[0]

        pbe1 = structures[0].shape[-1]
        pbe2 = structures[1].shape[-1]
        checkpoint = torch.load(pbe_model_name, map_location=torch.device(device))
        x = checkpoint['model_args']
        atom_fea_len = x['atom_fea_len']
        h_fea_len = x['h_fea_len']
        n_conv = x['n_conv']
        n_h = x['n_h']
        model_pbe = GCNE(pbe1,pbe2,atom_fea_len,n_conv,h_fea_len,n_h)
        model_pbe.cuda() if torch.cuda.is_available() else model_pbe.to(device)
        model_pbe.load_state_dict(checkpoint['state_dict'])
        model_pbe.eval()

        bandgap1 = structures[0].shape[-1]
        bandgap2 = structures[1].shape[-1]
        checkpoint = torch.load(bandgap_model_name, map_location=torch.device(device)) #, weights_only=True
        x = checkpoint['model_args']
        atom_fea_len = x['atom_fea_len']
        h_fea_len = x['h_fea_len']
        n_conv = x['n_conv']
        n_h = x['n_h']
        model_bandgap = GCNE(bandgap1,bandgap2,atom_fea_len,n_conv,h_fea_len,n_h)
        model_bandgap.cuda() if torch.cuda.is_available() else model_bandgap.to(device)
        model_bandgap.load_state_dict(checkpoint['state_dict'])
        model_bandgap.eval()
        
        for _, (input) in enumerate(loader):
            with torch.no_grad():
                input_var = (input[0].to(device),
                            input[1].to(device),
                            input[2].to(device),
                            input[3].to(device),
                            input[4].to(device),
                            input[5].to(device))
                pbe = model_pbe(*input_var)
                pbe = pbe_nor.denorm(pbe.data.cpu()).item()*num_atom
                bandgap = model_bandgap(*input_var)
                bandgap = bandgap_nor.denorm(bandgap.data.cpu()).item()
        
        print("PBE Energy: "+str(pbe)+" eV; "+"Band gap: "+str(bandgap)+" eV")
        return pbe, bandgap
    except Exception as e:
        print(e)
