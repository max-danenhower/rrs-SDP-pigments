import numpy as np
from rrsModelTrain import rrsModelTrain
import pandas as pd
import time

def train_model(RrsD, hplc):
    '''
    Define model hyper paramters and run the model. 

    Parameters:
    -----------
    RrsD: numpy array (n_wavelengths, n_samples)
        residual Rrs spectra
    hplc: numpy array (n_samples, n_pigments)
        HPLC measurments for the 13 pigments

    Saves A and C coefficients to an excel sheet
    '''

    # Use the 2nd derivative of the residual as model input
    diffD2 = np.diff(RrsD, 2, axis=0)

    n_permutations = 100 # the number of independent model validations to do (each formulates and validates a model)
    max_pcs = 30 # max number of spectral pc's to incorporate into the model - 30
    mdl_pick_metric = 'MAE' # pick the metric by which to evaluate pigment model fit - R2, RMSE, avg, med, ens, bias, MAE
    k = 5
    pft_index = 'pigment'

    pigs2mdl = np.array(['Tchla','Zea','DVchla','ButFuco','HexFuco','Allo','MVchlb',
                         'Neo','Viola','Fuco','Chlc12','Chlc3','Perid'])
    
    # variables specific to my HPLC dataset
    vars = (['Tchla','Tchlb','Tchlc','ABcaro','ButFuco','HexFuco','Allo','Diadino','Diato',
             'Fuco','Perid','Zea','MVchla','DVchla','Chllide','MVchlb','DVchlb','Chlc12','Chlc3',
             'Lut','Neo','Viola','Phytin','Phide','Pras'])
    
    start = time.time()

    summaries = []

    # create empty excel sheets with names a_coefs.xlsx and c_coefs.xlsx
    # before running
    with pd.ExcelWriter("a_coefs.xlsx", engine="openpyxl") as a_writer, \
        pd.ExcelWriter("c_coefs.xlsx", engine="openpyxl") as c_writer:
    
        # Start modelling
        for i in range(len(pigs2mdl)):
            pigment = pigs2mdl[i]
            pigment_index = vars.index(pigment)
            hplc_i = hplc[:, pigment_index]
            coefficients, intercepts, summary_gofs, all_gofs = rrsModelTrain(diffD2.T, hplc_i, pft_index,n_permutations, max_pcs, k, mdl_pick_metric) 
            summaries.append(summary_gofs)

            # save coefficients to xlsx
            pd.DataFrame(coefficients).to_excel(a_writer, sheet_name=pigment, index=False)
            pd.DataFrame(intercepts).to_excel(c_writer, sheet_name=pigment, index=False)

    print('Time taken:', time.time() - start)

