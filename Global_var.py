import os
import sys
import warnings
warnings.filterwarnings("ignore")
Path = os.path.dirname(os.path.abspath(__file__))
parser_dir = os.path.join(Path, "Parsers")
builder_dir = os.path.join(Path, "DataTrans")
work_dir = os.path.join(Path, "work")
model_dir = os.path.join(Path, "Model")
algorithm_dir = os.path.join(Path, "Prediction_Algorithm")

sys.path.append(parser_dir);sys.path.append(builder_dir);sys.path.append(work_dir);sys.path.append(model_dir); sys.path.append(algorithm_dir)

Lib_Path = os.path.join(work_dir, "Timing_Lib")
Itf_Path = os.path.join(work_dir, "ITF")
PtRpt_Path = os.path.join(work_dir, "PtRpt")
# PtScript_Path = os.path.join(work_dir, "PtScripts")

# Saved data path
Saved_Data_Path = os.path.join(Path, "DataTrans/Data")
Dataset_Path = os.path.join(Path, "DataTrans/Dataset")
Model_Path = os.path.join(Path, "Model/Trained_Model")

FEOL_Corners = []
for temp in ["m40", "0", "125"]:
    for volt in ["0p88v", "0p99v", "1p05v"]:
        FEOL_Corners.append(("ffg", volt, temp))
    for volt in ["0p72v", "0p81v", "0p9v"]:
        FEOL_Corners.append(("ssg", volt, temp))
for temp in ["85", "25"]:
    for volt in ["0p8v", "0p9v", "1v"]:
        FEOL_Corners.append(("tt", volt, temp))

#base_corner = ("ssg", "0p72v", "125", "cworst")
#base_corner = ("ffg", "1p05v", "m40", "cbest")
base_corner = ("ssg", "0p81v", "m40", "rcbest")
BEOL_Corners = ["rcbest", "rcworst", "cbest", "cworst", "typical"]

Designs = ["des", "mc_top", "ac97_top", "openGFX430", "wb_conmax_top", "eth_top", "vga_enh_top", "nova", "sasc_top", "spi_top", "tv80_core", "aes_cipher_top", "usbf_top", "pci_bridge32", "wb_dma_top", "tate_pairing"]

# TrainDesigns = ["des", "mc_top", "ac97_top", "wb_conmax_top", "eth_top", "openGFX430", "vga_enh_top", "nova"]
TrainDesigns = ["des", "mc_top", "ac97_top", "openGFX430", "wb_conmax_top", "eth_top", "vga_enh_top", "nova"]
# TrainDesigns = ["aes_cipher_top"]
# TestDesigns = ["tate_pairing"]
TestDesigns = ["sasc_top", "spi_top", "tv80_core", "aes_cipher_top", "usbf_top", "pci_bridge32", "wb_dma_top", "tate_pairing"]
