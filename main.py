import Kramer_hyperRrs
import Kramer_Rrs_pigments
import pandas as pd
import numpy as np

def main():
    data = pd.read_excel('HPLC_Rrs_forAli_2025.xlsx', header=0)

    sal = data.loc[:,'Sal']
    temp = data.loc[:,'Temp']
    Rrs = data.loc[:,'Rrs400':]

    rrsD, RrsD = Kramer_hyperRrs.get_rrs_residuals(Rrs, temp, sal)

    diff = np.diff(RrsD, 2, axis=0)

    # save diff to xlsx
    # pd.DataFrame(diff).to_excel('RrsD2.xlsx', index=False)
    # saved wavelength x sample (rows = wavelengths, columns = samples)

    hplc = data.loc[:,'Tchla':'Pras'].values

    Kramer_Rrs_pigments.train_model(RrsD, np.arange(400,701), hplc)



def run_python_coefs():
    a = pd.read_excel('python_a_coefs.xlsx', header=None)
    c = pd.read_excel('python_c_coefs.xlsx', header=None)
    drrs = pd.read_excel('dRrs_forAli_2025.xlsx')

    median_runs = np.zeros(len(drrs))

    for i in range(len(drrs)):
        spectra = drrs.iloc[i,:].values

        all_runs = np.zeros(100)

        for j in range(100):
            a_run = a.iloc[:,j].values
            c_run = c.iloc[j].values

            run = np.sum(a_run * spectra) + c_run
            all_runs[j] = run
        
        median_runs[i] = np.median(all_runs)

    # remove values below zero
    median_runs[median_runs < 0] = 0
    print(median_runs)

def run_matlab_coefs():
    a = pd.read_csv('tchla_hc.csv')
    c = pd.read_csv('tchla_hi.csv', header=None)
    drrs = pd.read_csv('hgsmD2.csv')

    median_runs = np.zeros(len(drrs))

    for i in range(len(drrs)):
        spectra = drrs.iloc[i,:].values
        all_runs = np.zeros(100)

        for j in range(100):
            a_run = a.iloc[:,j+1].values
            c_run = c.iloc[:,j].values

            run = np.sum(a_run * spectra) + c_run
            all_runs[j] = run[0]
        
        median_runs[i] = np.median(all_runs)

    # remove values below zero
    median_runs[median_runs < 0] = 0
    print(median_runs)

if __name__ == "__main__":
    main()
    #run_python_coefs()
    #run_matlab_coefs()