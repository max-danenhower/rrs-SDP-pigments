import numpy as np
from rrsModelTrain import rrsModelTrain
import pandas as pd
import time

def train_model(RrsD, wavelengths, hplc):
    diffD2 = np.diff(RrsD, 2, axis=0)

    n_permutations = 100
    max_pcs = 30
    mdl_pick_metric = 'MAE'
    k = 5
    pft_index = 'pigment'
    ofn_suffix = '-output.xlsx'

    pigs2mdl = np.array(['Tchla','Zea','DVchla','ButFuco','HexFuco','Allo','MVchlb',
                         'Neo','Viola','Fuco','Chlc12','Chlc3','Perid'])
    vars = (['Tchla','Tchlb','Tchlc','ABcaro','ButFuco','HexFuco','Allo','Diadino','Diato',
             'Fuco','Perid','Zea','MVchla','DVchla','Chllide','MVchlb','DVchlb','Chlc12','Chlc3',
             'Lut','Neo','Viola','Phytin','Phide','Pras'])
    
    start = time.time()

    summaries = []

    with pd.ExcelWriter("python_a_coefs.xlsx", engine="openpyxl") as a_writer, \
        pd.ExcelWriter("python_c_coefs.xlsx", engine="openpyxl") as c_writer:
    
        for i in range(len(pigs2mdl)):
            pigment = pigs2mdl[i]
            pigment_index = vars.index(pigment)
            pft = hplc[:, pigment_index]
            output_file_name = pigment + ofn_suffix
            coefficients, intercepts, summary_gofs, all_gofs = rrsModelTrain(diffD2.T, pft, pft_index,n_permutations, max_pcs, k, mdl_pick_metric, output_file_name) 
            summaries.append(summary_gofs)

            # save coefficients to xlsx
            
            pd.DataFrame(coefficients).to_excel(a_writer, sheet_name=pigment, index=False)
            pd.DataFrame(intercepts).to_excel(c_writer, sheet_name=pigment, index=False)

    print('Time taken:', time.time() - start)

