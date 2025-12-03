import Kramer_hyperRrs
import Kramer_Rrs_pigments
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

def generate_coefficients():
    '''
    Read in Rrs, temperature, and salinity data. 
    Define wavelengths corresponding to the Rrs spectra.

    Run Kramer_hyperRrs to get Rrs residuals.
    Run Kramer_Rrs_pigments to train the model.

    Running train_model will generate A and C coefficients. Make sure to create empty excel sheets to hold coefficients before running.
    See Kramer_Rrs_pigments.py for details. 
    '''

    data = pd.read_excel('HPLC_Rrs_forAli_2025.xlsx', header=0)

    sal = data.loc[:,'Sal'].values # array: (n_samples,)
    temp = data.loc[:,'Temp'].values # array: (n_samples,)
    Rrs = data.loc[:,'Rrs400':] # DataFrame: (n_samples, n_wavelengths) 
    wavelegnths = np.arange(400,701) # array: (n_wavelengths,)

    # get Rrs residuals
    rrsD, RrsD = Kramer_hyperRrs.get_rrs_residuals(Rrs, temp, sal, wavelegnths)

    hplc = data.loc[:,'Tchla':'Pras'].values

    # train model, create a spreadsheet with coefficients
    Kramer_Rrs_pigments.train_model(RrsD, hplc)
    
def run_sdp(rrs,wl,sst,sss):
    '''
    Method to show how to apply coefficients to an Rrs spectra to generate pigment values.
    
    pigment_concentration = sum(A(wavelength_i) * Rrs_residual(wavelength_i)) + C

    Coefficients are read from excel sheets created during the training process.

    Parameters:
    -----------

    rrs : DataFrame(n_samples, n_wavelengths)
        DataFrame where each row is an Rrs spectra with columns as wavelengths.
    wl : array-like(n_wavelengths)
        Wavelengths corresponding to the Rrs spectra columns.
    sst : array-like(n_samples)
        Sea surface temperature values corresponding to each Rrs spectra.
    sss : array-like(n_samples)
        Sea surface salinity values corresponding to each Rrs spectra.
    '''
 
    sdp_names = ['Tchla','MVchlb','Chlc12']

    rrs_residuals =  Kramer_hyperRrs.get_rrs_residuals(rrs, sst, sss, wl)[1]
    rrs_residuals_d2 = np.diff(rrs_residuals, 2, axis=0).T

    sdp = np.zeros((rrs_residuals_d2.shape[0],len(sdp_names)))

    for p, name in enumerate(sdp_names):

        # need to create excel sheets named 'a_coefs.xlsx' and 'c_coefs.xlsx' with appropriate sheets before running
        a_coefs = pd.read_excel('a_coefs.xlsx', sheet_name=name).values  # shape: (n_wl, 100)
        c_coefs = pd.read_excel('c_coefs.xlsx', sheet_name=name).values.flatten()  # shape: (100,)

        # Matrix multiplication to compute all runs at once for all samples
        # Result: run_vals_all shape (n_samples, 100)
        run_vals_all = rrs_residuals_d2 @ a_coefs + c_coefs

        # Take median over runs axis (axis=1)
        median_run = np.median(run_vals_all, axis=1)

        # Enforce non-negative
        median_run[median_run < 0] = 0

        sdp[:, p] = median_run

    return pd.DataFrame(sdp, columns=['T chla', 'MV chlb', 'chl c1+c2'])

def plot_pigments(data, lower_bound, upper_bound, label):
    '''
    Plots the pigment data from an L3 file with lat/lon coordinates using a color map

    Paramaters:
    -----------
    data : Xarray data array
        Contains pigment values to be plotted.
    lower_bound : float
        The lowest value represented on the color scale.
    upper_bound : float
        The upper value represented on the color scale.
    label : string
        A label for the graph.
    '''

    data.attrs["long_name"] = label

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, cmap.N))
    colors = np.vstack((np.array([1, 1, 1, 1]), colors)) 
    custom_cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(np.linspace(lower_bound, upper_bound, cmap.N)), ncolors=custom_cmap.N) 

    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels={"left": "y", "bottom": "x"})
    data.plot(cmap=custom_cmap, ax=ax, norm=norm)
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
    plt.show()
    
if __name__ == "__main__":
    generate_coefficients()
