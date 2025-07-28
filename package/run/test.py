from PACMANCharge import pmcharge
pmcharge.predict(cif_file="./Cu-BTC.cif",charge_type="DDEC6",digits=10,atom_type=True,neutral=True,keep_connect=True)
pmcharge.Energy(cif_file="./Cu-BTC.cif")

# from CifFile import ReadCif

# pacman_mof = ReadCif('Cu-BTC_pacman.cif')
# charge = pacman_mof[pacman_mof.keys()[0]]["_atom_site_charge"]

# mof = ReadCif('Cu-BTC.cif')
# mof.first_block().AddToLoop("_atom_site_type_symbol",{'_atom_site_charge':[ float(q) for q in charge]})
# print(type(charge))
# print(charge)
# name='cu'

# with open('mod.cif', 'w') as f:
#     # f.write(mof.WriteOut())
#     f.write("# charges by PACMAN v1.3 (https://github.com/mtap-research/PACMAN-charge/)\n" + f"data_{name}" + str(mof.first_block()))
