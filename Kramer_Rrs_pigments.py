import numpy as np
from rrsModelTrain import rrsModelTrain
import pandas as pd

def train_model(RrsD, wavelengths, hplc):
    diffD2 = np.diff(RrsD, 2)
    

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
    
    summaries = []
    
    for i in range(len(pigs2mdl)):
        pigment = pigs2mdl[i]
        print('pigment',pigment)
        pigment_index = vars.index(pigment)
        pft = hplc[1:len(hplc)-1, pigment_index]
        output_file_name = pigment + ofn_suffix
        coefficients, intercepts, summary_gofs, all_gofs = rrsModelTrain(diffD2.T, pft, pft_index,n_permutations, max_pcs, k, mdl_pick_metric, output_file_name) 
        summaries.append(summary_gofs)
    
    for s in summaries:
        print(s.iloc[0,:])