<h1 align="center">PACMAN</h1>

<h4 align="center">

</h4>              

A **P**artial **A**tomic **C**harge Predicter for Porous **Ma**terials based on Graph Convolutional Neural **N**etwork (**PACMAN**)   

[![Requires Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg?logo=python&logoColor=white)](https://python.org/downloads) [![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.10822403-blue)](https://doi.org/10.5281/zenodo.10822403)  [![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Chung-Research-Group/PACMAN-charge/LICENSE) [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sxmzhaogb@gmail.com) [![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)]() [![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)]()          


# Usage

```sh      
from PACMANCharge import pmcharge
pmcharge.predict(cif_file="./test/Cu-BTC.cif",charge_type="DDEC6",digits=6,atom_type=True,neutral=True,keep_connect=True)
pmcharge.Energy(cif_file="./test/Cu-BTC.cif")
```

* cif_file: cif file (without partial atomic charges) **[cif path]**                                                            
* charge-type (default: DDE6): DDEC6, Bader, CM5 or REPEAT                                         
* digits (default: 6): number of decimal places to print for partial atomic charges. ML models were trained on a 6-digit dataset.                                                                     
* atom-type (default: True): keep the same partial atomic charge for the same atom types (based on the similarity of partial atomic charges up to 2 decimal places).                                                         
* neutral (default: True): keep the net charge is zero. We use "mean" method to neuralize the system where the excess charges are equally distributed across all atoms.     
* keep_connect (default: True): retain the atomic and connection information (such as _atom_site_adp_type, bond) for the structure.                                                        

# Website & Zenodo
PACMAN-APP[link](https://pacman-charge-mtap.streamlit.app/)       
github repository[link](https://github.com/Chung-Research-Group/PACMAN-charge)                                                                          
DOWNLOAD full code and dataset[link](https://zenodo.org/records/10822403) But we will not update new vesion in Zenodo.            

# Reference
If you use PACMAN Charge, please cite this paper:
```
@article{       doi : 10.1021/acs.jctc.4c00434 ,
                author = {Zhao, Guobin and Chung, Yongchul G.},
                title = {PACMAN: A Robust Partial Atomic Charge Predicter for Nanoporous Materials Based on Crystal Graph Convolution Networks},
                journal = {Journal of Chemical Theory and Computation},
                volume = {20},
                number = {12},
                pages = {5368-5380},
                year = {2024},
                doi = {10.1021/acs.jctc.4c00434},
                note ={PMID: 38822793},
                URL = {https://doi.org/10.1021/acs.jctc.4c00434},
                eprint = {https://doi.org/10.1021/acs.jctc.4c00434}
        }
```

# Bugs

If you encounter any problem during using ***PACMAN***, please email ```sxmzhaogb@gmail.com```.                 

 
**Group:**   [Molecular Thermodynamics & Advance Processes Laboratory](https://sites.google.com/view/mtap-lab)                                
