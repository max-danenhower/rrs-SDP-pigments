import Kramer_hyperRrs
import Kramer_Rrs_pigments
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

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

    print(rrs_residuals_d2)
    print(rrs_residuals_d2.shape)

    sdp = np.zeros((rrs_residuals_d2.shape[0],len(sdp_names)))

    for p, name in enumerate(sdp_names):
        print(name)

        a_coefs = pd.read_excel('python_a_coefs.xlsx', sheet_name=name).values  # shape: (n_wl, 100)
        c_coefs = pd.read_excel('python_c_coefs.xlsx', sheet_name=name).values.flatten()  # shape: (100,)

        # Matrix multiplication to compute all runs at once for all samples
        # Result: run_vals_all shape (n_samples, 100)
        run_vals_all = rrs_residuals_d2 @ a_coefs + c_coefs

        # Take median over runs axis (axis=1)
        median_run = np.median(run_vals_all, axis=1)

        # Enforce non-negative
        median_run[median_run < 0] = 0

        sdp[:, p] = median_run

    return pd.DataFrame(sdp, columns=['T chla', 'MV chlb', 'chl c1+c2'])

def create_dataset(rrs_paths, sal_paths, temp_paths, bbox):
    '''
    Creates an xarray data array with latitude and longitude coordinates. Each coordinate contains a hyperspectral Rrs spectra with 
    corresponding wavelenghts, salinity, and temperature. If more than one file for Rrs, salinity, or temperature are given, uses the 
    date averaged values. 

    Parameters:
    -----------
    rrs_paths : list or str
        A single file path to a PACE Rrs file or a list of file paths to PACE Rrs files.
    sal_paths : list or str
        A single file path to a salinity file or a list of file paths to salinity files.
    temp_paths : list or str
        A single file path to a temperature file or a list of file paths to temperature files.
    bbox : tuple of floats or ints
        A tuple representing spatial bounds in the form (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).

    Returns:
    --------
    xarray.Dataset
        A data array of Rrs values at each wavelength over a specified lat/lon box.

    Raises:
    -------
    TypeError 
        If rrs_paths, sal_paths, or temp_paths is not a string or list.
    '''

    n = bbox[3]
    s = bbox[1]
    e = bbox[2]
    w = bbox[0]
    
    # creates a dataset of rrs values of the given file
    if isinstance(rrs_paths, str):
        rrs_data = xr.open_dataset(rrs_paths)
        rrs = rrs_data["Rrs"].sel({"lat": slice(n, s), "lon": slice(w, e)})
    elif isinstance(rrs_paths, list):
        # if given a list of files, create a date averaged dataset of Rrs values 
        rrs_data = xr.open_mfdataset(
            rrs_paths,
            combine="nested",
            concat_dim="date",
        )
        rrs = rrs_data["Rrs"].sel({"lat": slice(n, s), "lon": slice(w, e)}).mean('date')
        rrs = rrs.compute()
    else:
        raise ValueError('rrs_paths must be a string or a list containing at least one filepath')
    
    # creates a dataset of sal and temp values of the given file
    if isinstance(sal_paths, str):
        sal = xr.open_dataset(sal_paths)
        sal = sal["smap_sss"].interp(longitude=rrs.lon, latitude=rrs.lat, method='nearest')
    elif isinstance(sal_paths, list):
        # if given a list of files, create a date averaged dataset of salinity values 
        sal = xr.open_mfdataset(
            sal_paths,
            combine="nested",
            concat_dim="date",
        )
        sal = sal["smap_sss"].interp(longitude=rrs.lon, latitude=rrs.lat, method='nearest').mean('date')
        sal = sal.compute()
    else:
        raise TypeError('temp_paths must be a string or list')
    
    # creates a dataset of sal and temp values of the given file
    if isinstance(temp_paths, str):
        temp = xr.open_dataset(temp_paths)
        temp = temp['analysed_sst'].squeeze() # get rid of extra time dimension
        temp = temp.interp(lon=rrs.lon, lat=rrs.lat, method='nearest')
    elif isinstance(temp_paths, list):
        # if given a list of files, create a date averaged dataset of temperature values 
        temp = xr.open_mfdataset(
            temp_paths,
            combine="nested",
            concat_dim="time"
        )
        temp = temp['analysed_sst'].interp(lon=rrs.lon, lat=rrs.lat, method='nearest').mean('time')
        temp = temp.compute()
    else:
        raise TypeError('temp_paths must be a string or list')
    
    rrs = rrs.interp(wavelength=np.arange(400,701,1))

    return rrs, sal, temp

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

def sdp_from_pace(pace_file, sss, sst):

    rrs_interp, sss_interp, sst_interp = create_dataset(pace_file, sss, sst, (-166, -26, -165, -25))

    print(rrs_interp)
    wl = np.arange(400,701,1)

    rrs_np = rrs_interp.to_numpy().reshape(-1, rrs_interp.wavelength.size)
    rrs_df = pd.DataFrame(rrs_np, columns=wl)
    sss_np = sss_interp.to_numpy().flatten()
    sst_np = sst_interp.to_numpy().flatten()

    sdp = run_sdp(rrs_df, wl, sst_np, sss_np)

    chla = sdp['T chla'].values.reshape(rrs_interp.lat.size, rrs_interp.lon.size)
    chlb = sdp['MV chlb'].values.reshape(rrs_interp.lat.size, rrs_interp.lon.size)
    chlc = sdp['chl c1+c2'].values.reshape(rrs_interp.lat.size, rrs_interp.lon.size)

    pigments = xr.Dataset(
        {
            'chla': (['lat', 'lon'], chla),
            'chlb': (['lat', 'lon'], chlb),
            'chlc': (['lat', 'lon'], chlc),
        },
        coords={
            'lat': rrs_interp.lat.values,
            'lon': rrs_interp.lon.values,
        },
    )
    
if __name__ == "__main__":
    pace_file = 'pace_files/PACE_OCI.20240612.L3m.DAY.RRS.V3_0.Rrs.4km.nc'
    sss = 'pace_files/SMAP_L3_SSS_20240608_8DAYS_V5.0.nc'
    sst = 'pace_files/20240612090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'

    sdp_from_pace(pace_file, sss, sst)
