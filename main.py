import Kramer_hyperRrs
import Kramer_Rrs_pigments
import pandas as pd
import numpy as np

def main():
    '''
    Read in Rrs, temperature, and salinity data. 
    Define wavelengths corresponding to the Rrs spectra.

    Run Kramer_hyperRrs to get Rrs residuals.
    Run Kramer_Rrs_pigments to run train the model.
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

def run_python_coefs():
    '''
    Method to show how to apply coefficients to an Rrs spectra to generate pigment values.

    Required inputs: A coefficients, C coefficients, Rrs residual
    
    pigment_concentration = sum(A(wavelength_i) * Rrs_residual(wavelength_i)) + C
    '''

    a = pd.read_excel('python_a_coefs.xlsx', header=None)
    c = pd.read_excel('python_c_coefs.xlsx', header=None)
    drrs = pd.read_excel('dRrs_forAli_2025.xlsx')

    median_runs = np.zeros(len(drrs))

    for i in range(len(drrs)):
        # for each spectra 

        spectra = drrs.iloc[i,:].values

        all_runs = np.zeros(100)

        for j in range(100):
            # for each run (here there are 100 runs)

            a_run = a.iloc[:,j].values
            c_run = c.iloc[j].values

            run = np.sum(a_run * spectra) + c_run
            all_runs[j] = run
        
        # use the median of all runs as the pigment value
        median_runs[i] = np.median(all_runs)

    # remove values below zero
    median_runs[median_runs < 0] = 0
    print(median_runs)

if __name__ == "__main__":
    main()
    #run_python_coefs()
