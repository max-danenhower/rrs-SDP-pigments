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

if __name__ == "__main__":
    main()