import Kramer_hyperRrs
import Kramer_Rrs_pigments
import pandas as pd
import numpy as np

def main():
    data = pd.read_csv('EXP_NP_NA_HPLC_Rrs.csv', header=0)

    sal = data.loc[:,'Salinity']
    temp = data.loc[:,'Temperature']
    Rrs = data.loc[:,'Rrs400':]
    chl = data.loc[:,'Tchla (mg m^-3)':'Pras']

    rrsD, RrsD = Kramer_hyperRrs.get_rrs_residuals(Rrs, temp, sal, chl)

    hplc = data.loc[:,'Tchla (mg m^-3)':'Pras'].values

    Kramer_Rrs_pigments.train_model(RrsD, np.arange(400,701), hplc)

def run_coefs():
    a = pd.read_excel('python_a_coefs.xlsx', header=None)
    c = pd.read_excel('python_c_coefs.xlsx', header=None)
    drrs = pd.read_excel('dRrs_forAli_2025.xlsx')

    for i in range(len(drrs)):
        spectra = drrs.iloc[i,:].values

        all_runs = np.zeros(100)

        for j in range(100):
            a_run = a.iloc[:,j].values
            c_run = c.iloc[j].values

            run = np.sum(a_run * spectra + c_run)
            all_runs[j] = run
        
        print('median', np.median(all_runs))

if __name__ == "__main__":
    #main()
    run_coefs()