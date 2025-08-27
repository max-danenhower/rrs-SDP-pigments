import Kramer_hyperRrs
import Kramer_Rrs_pigments
import pandas as pd
import numpy as np

def main():
    '''
    Read in Rrs, temperature, and salinity data. 
    Define wavelengths corresponding to the Rrs spectra.

    Run Kramer_hyperRrs to get Rrs residuals.
    Run Kramer_Rrs_pigments to train the model.

    Running train_model will generate A and C coefficients. Make sure to create empty excel sheets to hold coeddicients before running.
    See Kramer_Rrs_pigments.py for details. 
    '''

    data = pd.read_excel('HPLC_Rrs_forAli_2025.xlsx', header=0)

    sal = data.loc[:,'Sal'].values
    temp = data.loc[:,'Temp'].values
    Rrs = data.loc[:,'Rrs400':]
    wavelegnths = np.arange(400,701)

    # get Rrs residuals
    rrsD, RrsD = Kramer_hyperRrs.get_rrs_residuals(Rrs, temp, sal, wavelegnths)

    hplc = data.loc[:,'Tchla':'Pras'].values

    # train model
    Kramer_Rrs_pigments.train_model(RrsD, hplc)
    
def run_sdp(rrs,wl,sst,sss):
    '''
    Method to show how to apply coefficients to an Rrs spectra to generate pigment values.

    Required inputs: A coefficients, C coefficients (read from excel sheets), Rrs spectra, wavelengths, temperature, salinity
    
    pigment_concentration = sum(A(wavelength_i) * Rrs_residual(wavelength_i)) + C
    '''
 
    sdp_names = ['Tchla','MVchlb','Chlc12']

    rrs_residuals =  Kramer_hyperRrs.get_rrs_residuals(rrs, sst, sss, wl)[1]
    rrs_residuals_d2 = np.diff(rrs_residuals, 2, axis=0).T

    sdp = np.zeros((rrs_residuals_d2.shape[0],len(sdp_names)))

    for p, name in enumerate(sdp_names):
        print(name)

        a_coefs = pd.read_excel('sdp_coefs/python_a_coefs.xlsx', sheet_name=name).values  # shape: (n_wl, 100)
        c_coefs = pd.read_excel('sdp_coefs/python_c_coefs.xlsx', sheet_name=name).values.flatten()  # shape: (100,)

        # Matrix multiplication to compute all runs at once for all samples
        # Result: run_vals_all shape (n_samples, 100)
        run_vals_all = rrs_residuals_d2 @ a_coefs + c_coefs

        # Take median over runs axis (axis=1)
        median_run = np.median(run_vals_all, axis=1)

        # Enforce non-negative
        median_run[median_run < 0] = 0

        sdp[:, p] = median_run

    return pd.DataFrame(sdp, columns=['T chla', 'MV chlb', 'chl c1+c2'])

if __name__ == "__main__":
    main()
    #run_sdp()
