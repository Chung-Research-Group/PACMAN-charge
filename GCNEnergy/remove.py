import shutil
import os
import glob
import re
import json

def move_to_here(json_file,folder):
    with open(json_file,'rb') as f:
        data = json.load(f)
    f.close()
    for cif in data:
        shutil.copy(data[cif][0],folder)

def reomove_occupancy(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    start_reading_charges = False
    new_lines = []
    for line in lines:
        if start_reading_charges:
            parts = [part for part in re.split(r'\s+', line) if part != '']
            if len(parts)>7:
                # print(parts)
                new_lines.append(line)
            else:
                pass
        else:
            new_lines.append(line)
        if "_atom_site_charge" in line:
            start_reading_charges = True
    
    with open(path.replace(".cif","_fix.cif"), 'w') as file:
        file.writelines(new_lines)
    file.close()

move_to_here("fail.json","other_0/")
move_to_here("fail1.json","other_0/")

mofs = glob.glob(os.path.join("other_0", '*.cif'))
for mof in mofs:
    reomove_occupancy(mof)