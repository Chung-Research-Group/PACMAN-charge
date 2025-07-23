import sys
import os
import glob
import json
import torch
import pickle
import importlib
from tqdm import tqdm
import pandas as pd
from model4pre.GCN_E import GCN
from model4pre.data import collate_pool, get_data_loader, CIFData
from model4pre.cif2data import CIF2json, n_atom

source = importlib.import_module('model4pre')
sys.modules['source'] = source

def main():
    if len(sys.argv) != 2:
        print("Usage: python GCNEnergy.py folder")
        sys.exit(1)
    path = sys.argv[1]
    if os.path.isfile(path):
        print("please input a folder, not a file")
    elif os.path.isdir(path):
        pass
    else:
        print("Can not find your file, please check is it exit or correct?")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_pbe_name = "./pth/best_pbe/pbe-atom.pth"
    model_bandgap_name = "./pth/best_bandgap/bandgap.pth"
    pbe_nor_name = "./pth/best_pbe/normalizer-pbe.pkl"
    bandgap_nor_name = "./pth/best_bandgap/normalizer-bandgap.pkl"
    
    with open(pbe_nor_name, 'rb') as f:
        pbe_nor = pickle.load(f)
    f.close()
    with open(bandgap_nor_name, 'rb') as f:
        bandgap_nor = pickle.load(f)
    f.close()

    cif_files = glob.glob(os.path.join(path, '*.cif'))
    print("# of atoms, PBE energy (eV), Bandgap (eV): ")

    data4csv = []
    dic = {}
    fail = {}
    i = 0
    for path in tqdm(cif_files):
        try:
            json_data = CIF2json(path)
            num_atom = n_atom(path)
            batch_size = 1
            num_workers = 0
            pin_memory = False
            pre_dataset = CIFData(path,json_data)
            collate_fn = collate_pool
            pre_loader= get_data_loader(pre_dataset,collate_fn,batch_size,num_workers,pin_memory)
            structures, _ = pre_dataset[0]
            pbe1 = structures[0].shape[-1]
            pbe2 = structures[1].shape[-1]
            checkpoint = torch.load(model_pbe_name, map_location=torch.device(device))
            x = checkpoint['model_args']
            atom_fea_len = x['atom_fea_len']
            h_fea_len = x['h_fea_len']
            n_conv = x['n_conv']
            n_h = x['n_h']
            model_pbe = GCN(pbe1,pbe2,atom_fea_len,n_conv,h_fea_len,n_h)
            model_pbe.cuda() if torch.cuda.is_available() else model_pbe.to(device)
            model_pbe.load_state_dict(checkpoint['state_dict'])
            model_pbe.eval()
            bandgap1 = structures[0].shape[-1]
            bandgap2 = structures[1].shape[-1]
            checkpoint = torch.load(model_bandgap_name, map_location=torch.device(device))
            x = checkpoint['model_args']
            atom_fea_len = x['atom_fea_len']
            h_fea_len = x['h_fea_len']
            n_conv = x['n_conv']
            n_h = x['n_h']
            model_bandgap = GCN(bandgap1,bandgap2,atom_fea_len,n_conv,h_fea_len,n_h)
            model_bandgap.cuda() if torch.cuda.is_available() else model_bandgap.to(device)
            model_bandgap.load_state_dict(checkpoint['state_dict'])
            model_bandgap.eval()
            for _, (input,cif_ids) in enumerate(pre_loader):
                with torch.no_grad():
                    if device == "cuda":
                        input_cuda = [input_tensor.to(device) for input_tensor in input]
                        input_var = (input_cuda[0].cuda(),
                                    input_cuda[1].cuda(),
                                    input_cuda[2].cuda(),
                                    input_cuda[3].cuda(),
                                    input_cuda[4].cuda(),
                                    input_cuda[5].cuda())
                    else:
                        input_var = (input[0],
                                    input[1],
                                    input[2],
                                    input[3],
                                    input[4],
                                    input[5])
                    pbe = model_pbe(*input_var)
                    pbe = pbe_nor.denorm(pbe.data.cpu()).item()*num_atom
                    bandgap = model_bandgap(*input_var)
                    bandgap = bandgap_nor.denorm(bandgap.data.cpu()).item()
                    print(cif_ids[0] + ": " + str(num_atom) +"," + str(pbe) + "," + str(bandgap) + " / ev")
                    data4csv.append([cif_ids[0],num_atom,pbe,bandgap])
                    dic[cif_ids[0]] = [num_atom,pbe,bandgap]
                    i+=1
        except:
            print("Fail predict: " + path)
            fail[str(i)]=[path]
            i += 1
        with open("./preE.json",'w') as f:
            json.dump(dic,f)
        f.close()
        with open("fail.json",'w') as f:
            json.dump(fail,f)
        f.close()
        df_data4csv = pd.DataFrame(data4csv,columns=["name","natoms","PBE","Bandgap"])
        df_data4csv.to_csv("pre_PBE_Bandgap.csv",index=False)

if __name__ == "__main__":
    main()
