import linecache
import re
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var

class Delay:
    def __init__(self):
        self.index1 = [] # input transition
        self.index2 = [] # output capacitance
        self.delay = []
    def __repr__(self):
        delay_str = "\n".join(map(str, self.delay[0:]))
        return f"\nindex1:\n{self.index1}\nindex2:\n{self.index2}\nDelay:\n{delay_str}\n"

class Trans:
    def __init__(self):
        self.index1 = []
        self.index2 = []
        self.trans = []
        self.isscalar = False # for specific type of cell with scalar type
    def __repr__(self):
        trans_str = "\n".join(map(str, self.trans[0:]))
        return f"\nindex1:\n{self.index1}\nindex2:\n{self.index2}\nTrans:\n{trans_str}\n"

class Constrain:
    def __init__(self):
        self.index1 = []
        self.index2 = []
        self.constrain = []
    def __repr__(self):
        Constrain_str = "\n".join(map(str, self.constrain[0:]))
        return f"\nindex1:\n{self.index1}\nindex2:\n{self.index2}\nConstrain:\n{Constrain_str}\n"

class Cell_Timing:
    def __init__(self):
        self.delay = {}
        self.trans = {}
        self.constrain = {} # constrained time
        self.name = ''
        self.footprint = ''
        self.pins = []
    def __repr__(self):
        return f"CellName: {self.name}\npins:{self.pins}\n{self.delay}\n{self.trans}\n{self.constrain}"

def Read_TimingLib(inlib):
    cells = {}
    if not os.path.exists(inlib):
        raise FileNotFoundError(f"Timing library file not found: {inlib}")
    
    with open(inlib, 'r') as infile:
        linecount = 0
        for line in infile:
            linecount += 1
            index = line.split()
            # cell insertation
            if(len(index) >= 3):
                if(index[0] == 'cell' and '(' in index[1] and ')' in index[1] and index[2] == '{'):
                    try:
                        newcell
                    except NameError:
                        pass
                    else:
                        cells[newcell.name] = newcell
                    # new cell found
                    newcell = Cell_Timing()
                    newcell.name = index[1].replace('(', '').replace(')', '')
                    #cell_names.append(index[1].replace('(', '').replace(')', ''))
            if(len(index) >= 3):
                # footprint
                if(index[0] == 'cell_footprint'):
                    newcell.footprint = index[2].replace('"', '').replace(';', '')
            if(len(index) >= 2):
                # new pin found
                if(index[0] == 'pin' and '(' in index[1] and ')' in index[1] and index[2] == '{') or (index[0].find('pin(') != -1):
                    if len(index) == 3:
                        newcell.pins.append(index[1].replace('(', '').replace(')', ''))
                    else:
                        newcell.pins.append(index[0].replace('pin(', '').replace(')', ''))
                    next_line = linecache.getline(inlib, linecount + 1)
                    next_index = next_line.split()
                    if(len(next_index) >= 3):
                        # outpin
                        if(next_index[0].find('direction') != -1 and next_index[2].find('output') != -1) or (next_index[0].find('clock_gate_out_pin') != -1 and next_index[2].find('true') != -1):
                            if len(index) == 3:
                                outpin = index[1].replace('(', '').replace(')', '')
                            elif len(index) == 2:
                                outpin = index[0].replace('pin(', '').replace(')', '')
                # timing info
                if(index[0] == 'timing'):
                    next_line = linecache.getline(inlib, linecount + 1)
                    next_index = next_line.split()
                    if(len(next_index) >= 3):
                        if(next_index[0] == 'related_pin'):
                            inpin = next_index[2].replace('"', '').replace(';', '')
                if(index[0] == 'cell_rise'):
                    rf = 'r'
                elif(index[0] == 'cell_fall'):
                    rf = 'f'
                if(index[0] == 'cell_rise' or index[0] == 'cell_fall'):
                    newdelay = Delay()
                    arc = inpin + '->' + outpin
                    template = re.findall(r'\d+', index[1])
                    index1_length = int(template[0])
                    index1_line = linecache.getline(inlib, linecount + 1)
                    index1_index = index1_line.split()
                    for i in range(index1_length):
                        newdelay.index1.append(float(index1_index[1+i].replace('(', '').replace('"', '').replace(')', '').replace(';', '').replace(',', '')))
                    index2_length = int(template[1])
                    index2_line = linecache.getline(inlib, linecount + 2)
                    index2_index = index2_line.split()
                    for i in range(index2_length):
                        newdelay.index2.append(float(index2_index[1+i].replace('(', '').replace('"', '').replace(')', '').replace(';', '').replace(',', '')))
                    for j in range(index2_length):
                        delay_line = linecache.getline(inlib, linecount + j + 4)
                        delay_index = delay_line.split()
                        delays = []
                        for i in range(index1_length):
                            delays.append(float(delay_index[i].replace('"', '').replace(',', '')))
                        newdelay.delay.append(delays)
                    newcell.delay[(arc,rf)] = newdelay
                # transition info
                if(index[0] == 'rise_transition'):
                    rf = 'r'
                elif(index[0] == 'fall_transition'):
                    rf = 'f'
                if(index[0] == 'rise_transition' or index[0] == 'fall_transition'):
                    newtrans = Trans()
                    arc = inpin + '->' + outpin
                    if(index[1] == '(scalar)'):
                        newtrans.index1 = None
                        newtrans.index2 = None
                        newtrans.isscalar = True
                        newtrans.trans = 0
                        continue
                    template = re.findall(r'\d+', index[1])
                    index1_length = int(template[0])
                    index1_line = linecache.getline(inlib, linecount + 1)
                    index1_index = index1_line.split()
                    for i in range(index1_length):
                        newtrans.index1.append(float(index1_index[1+i].replace('(', '').replace('"', '').replace(')', '').replace(';', '').replace(',', '')))
                    index2_length = int(template[1])
                    index2_line = linecache.getline(inlib, linecount + 2)
                    index2_index = index2_line.split()
                    for i in range(index2_length):
                        newtrans.index2.append(float(index2_index[1+i].replace('(', '').replace('"', '').replace(')', '').replace(';', '').replace(',', '')))
                    for j in range(index2_length):
                        trans_line = linecache.getline(inlib, linecount + j + 4)
                        trans_index = trans_line.split()
                        transes = []
                        for i in range(index1_length):
                            transes.append(float(trans_index[i].replace('"', '').replace(',', '')))
                        newtrans.trans.append(transes)
                    newcell.trans[(arc,rf)] = newtrans
                if(index[0] == 'timing_type'):
                    if(index[2] == 'setup_rising;' or index[2] == 'setup_falling;' or index[2] == 'hold_rising;' or index[2] == 'hold_falling;'):
                        for i in range(1, 20): # check next 20 lines
                            count = linecount + i
                            template_line = linecache.getline(inlib, count)
                            template_index = template_line.split()
                            if(template_index[0].find('constraint') != -1):
                                newconstrain = Constrain()
                                template = re.findall(r'\d+', template_index[1])
                                index1_length = int(template[0])
                                index1_line = linecache.getline(inlib, count + 1)
                                index1_index = index1_line.split()
                                for i in range(index1_length):
                                    newconstrain.index1.append(float(index1_index[1+i].replace('(', '').replace('"', '').replace(')', '').replace(';', '').replace(',', '')))
                                index2_length = int(template[1])
                                index2_line = linecache.getline(inlib, count + 2)
                                index2_index = index2_line.split()
                                for i in range(index2_length):
                                    newconstrain.index2.append(float(index2_index[1+i].replace('(', '').replace('"', '').replace(')', '').replace(';', '').replace(',', '')))
                                for j in range(index2_length):
                                    constrain_line = linecache.getline(inlib, count + 4 + j)
                                    constrain_index = constrain_line.split()
                                    constrains = []
                                    for i in range(index1_length):
                                        constrains.append(float(constrain_index[i].replace('"', '').replace(',', '')))
                                    newconstrain.constrain.append(constrains)
                                if(index[2].find('setup') != -1):
                                    if(template_index[0].find('rise') != -1):
                                        newcell.constrain[('setup', 'r')] = newconstrain
                                    if(template_index[0].find('fall') != -1):
                                        newcell.constrain[('setup', 'f')] = newconstrain
                                if(index[2].find('hold') != -1):
                                    if(template_index[0].find('rise') != -1):
                                        newcell.constrain[('hold', 'r')] = newconstrain
                                    if(template_index[0].find('fall') != -1):
                                        newcell.constrain[('hold', 'f')] = newconstrain
        cells[newcell.name] = newcell
        print(f"{inlib.split('/')[-1]} Readed.")
    return cells

import pickle
if __name__ == "__main__":
    lib_dir = os.path.join(Global_var.work_dir, "Timing_Lib")
    inlib = os.path.join(lib_dir, "tcbn28hpcplusbwp7t40p140ssg0p72v125c_ccs.lib")
    cells = Read_TimingLib(inlib)
    with open("./1", "wb") as f:
       pickle.dump(cells ,f) 
    # for cellname, cell_Timing in cells.items():
    #     print(cellname)
    #     print(cell_Timing.pins)