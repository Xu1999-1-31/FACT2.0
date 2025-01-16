import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var

class METAL:
    def __init__(self):
        self.name = ''
        self.thickness = 0
        self.e_above = 0
        self.e_below = 0
    def __repr__(self):
        return f"Metal: {self.name} thickness: {self.thickness} e_above: {self.e_above} e_below: {self.e_below}\n"

class Dielectric:
    def __init__(self):
        self.name = ''
        self.thickness = 0
        self.e = 0
    def __repr__(self):
        return f"Dielectric: {self.name} thickness: {self.thickness} e: {self.e}\n"
    
def Read_Itf(initf):
    if not os.path.exists(initf):
        raise FileNotFoundError(f"ITF file not found: {initf}")
    metals, layers = {}, []
    with open(initf, 'r') as infile:
        for line in infile:
            index = line.split()
            if(len(index) > 0):
                if(index[0] == 'CONDUCTOR'):
                    if(index[2].find('THICKNESS') != -1):
                        new_metal = METAL()
                        new_metal.name = index[1]
                        new_metal.thickness = float(index[3])
                        layers.append(new_metal)
                        metals[new_metal.name] = new_metal
                    else:
                        break
                if(index[0] == 'DIELECTRIC'):
                    new_dielec = Dielectric()
                    new_dielec.name = index[1]
                    new_dielec.thickness = float(index[2].split('=')[1])
                    new_dielec.e = float(index[3].split('=')[1])
                    layers.append(new_dielec)

    for metal in metals.values():
        sigma_e = 0
        sigma_d = 0
        for layer in layers:
            if(isinstance(layer, METAL) and layer.name == metal.name):
                metal.e_above = sigma_d/sigma_e
                sigma_e = 0
                sigma_d = 0
            if(isinstance(layer, Dielectric)):
                sigma_e += layer.thickness/layer.e
                sigma_d += layer.thickness
        metal.e_below = sigma_d/sigma_e
    return metals

if __name__ == "__main__":
    metals = Read_Itf("cln28hpc+_1p09m+ut-alrdl_4x2y2r_rcworst.itf")
    print(metals)