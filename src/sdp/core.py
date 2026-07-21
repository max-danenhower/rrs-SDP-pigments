from datetime import datetime
from importlib.resources import files
from time import perf_counter
from pathlib import Path

import cartopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .Kramer_hyperRrs import get_rrs_residuals
from .Kramer_Rrs_pigments import train_model

ASSETS = files().joinpath('resources')


def generate_coefficients():
    '''
    Read in Rrs, temperature, and salinity data. 
    Define wavelengths corresponding to the Rrs spectra.

    Run Kramer_hyperRrs to get Rrs residuals.
    Run Kramer_Rrs_pigments to train the model.

    Running train_model will generate A and C coefficients. Make sure to create empty excel sheets to hold coefficients before running.
    See Kramer_Rrs_pigments.py for details. 
    '''

    data = pd.read_excel(ASSETS / 'HPLC_Rrs_forAli_2025.xlsx', header=0)

    sal = data.loc[:,'Sal'].values # array: (n_samples,)
    temp = data.loc[:,'Temp'].values # array: (n_samples,)
    Rrs = data.loc[:,'Rrs400':] # DataFrame: (n_samples, n_wavelengths) 
    wavelegnths = np.arange(400,701) # array: (n_wavelengths,)

    # get Rrs residuals
    rrsD, RrsD = get_rrs_residuals(Rrs, temp, sal, wavelegnths)

    hplc = data.loc[:,'Tchla':'Pras'].values

    # train model, create a spreadsheet with coefficients
    train_model(RrsD, hplc)
    
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
 
    sdp_names = [
        'Tchla',
        'MVchlb',
        'Chlc12',
        'Zea',
        'DVchla',
        'ButFuco',
        'HexFuco',
        'Allo',
        'Neo',
        'Viola',
        'Fuco',
        'Chlc3',
        'Perid'
    ]

    res_start = perf_counter()
    
    smoothed_rrs = (
        rrs.T
        .rolling(window=5, min_periods=1)
        .mean()
        .T
    )

    cutoff_rrs = smoothed_rrs.loc[:,400:700]


    print('calculating residuals')
    rrs_residuals = get_rrs_residuals(cutoff_rrs, sst, sss, wl)[1]
    res_end = perf_counter()
    print('residuals calculated', res_end-res_start)

    print('calculating 2nd derivative')
    deriv_start = perf_counter()
    rrs_residuals_d2 = np.diff(rrs_residuals, 2, axis=0).T
    deriv_end = perf_counter()
    print('2nd derivative calculated', deriv_end-deriv_start)
    sdp = np.zeros((rrs_residuals_d2.shape[0],len(sdp_names)))

    np.save('rrs_residuals_d2.npy', rrs_residuals_d2)

    print(rrs_residuals_d2.shape)

    print('running coefs')
    coef_start = perf_counter()
    for p, name in enumerate(sdp_names):

        # Read in A and C coefficients
        # a_coefs shape: (n_wl, 100), c_coefs shape: (100,)
        # A and C coefficients need to be pre-computed and stored in excel sheets
        a_coefs = pd.read_excel(ASSETS / 'sdp_coefs' / 'original_a_coefs.xlsx', sheet_name=name, header=0)
        a_coefs = a_coefs.drop(columns=a_coefs.columns[0]).values # shape: (n_wl, 100)
        c_coefs = pd.read_excel(ASSETS / 'sdp_coefs' / 'original_c_coefs.xlsx', sheet_name=name, header=0).values.flatten()  # shape: (100,)

        # Matrix multiplication to compute all runs for all samples
        # Result: run_vals_all shape (n_samples, 100)
        run_vals_all = rrs_residuals_d2 @ a_coefs + c_coefs

        # Take median over runs axis (axis=1)
        median_run = np.median(run_vals_all, axis=1)

        # Enforce non-negative
        median_run[median_run < 0] = 0

        sdp[:, p] = median_run
    coef_end = perf_counter()
    print('coefs run complete', coef_end-coef_start)

    return pd.DataFrame(sdp, columns=sdp_names)

def interpolate_coords(rrs_path, sal_path, temp_path, bbox):
    '''
    Interpolate the salinity and temperature data coordinates onto the PACE L2 Rrs coordinates. Default use climatology files.
    Can pass in GHRSST and SMAP files as well. 

    Parameters:
    -----------
    rrs_path : str
        A single file path to a PACE L2 AOP file.
    sal_path : str
        A single file path to a salinity file.
    temp_path : str
        A single file path to a temperature file.
    bbox: tuple of floats or ints
        A tuple representing spatial bounds in the form (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).
        Default is None, which means the entire L2 granule will be processed

    Returns:
    --------
    rrs_box, rrs_unc_box, wavelength_coords, sal, temp : Xarrays all on the same lat/lon coordinates (except wavelength_coords, which is a 1D array)
    '''

    # Open PACE groups
    sensor_band_params = xr.open_dataset(rrs_path, group="sensor_band_parameters")
    geophysical_data = xr.open_dataset(rrs_path, group="geophysical_data")
    navigation_data = xr.open_dataset(rrs_path, group="navigation_data")
    
    # Extract wavelength coordinates
    wavelength = sensor_band_params["wavelength_3d"].values
    
    # Get Rrs and attach coordinates
    navigation_data = navigation_data.set_coords(("latitude", "longitude"))
    
    rrs = (
        geophysical_data["Rrs"]
        .assign_coords(
            latitude=navigation_data.latitude,
            longitude=navigation_data.longitude,
            wavelength=wavelength,
        )
    )
    
    if bbox is not None:
        w,s,e,n = bbox[0],bbox[1],bbox[2],bbox[3]

        try:
            rrs_box = rrs.where(
                (
                    (rrs["latitude"] > s)
                    & (rrs["latitude"] < n)
                    & (rrs["longitude"] < e)
                    & (rrs["longitude"] > w)
                ),
                drop=True,
            )
        except ValueError:
            print('Boundary box is outside of the granule boundary.')
            raise
    else:
        n = rrs.latitude.values.max()
        s = rrs.latitude.values.min() 
        e = rrs.longitude.values.max()
        w = rrs.longitude.values.min()

        rrs_box = rrs.where(
            (
                (rrs["latitude"] > s)
                & (rrs["latitude"] < n)
                & (rrs["longitude"] < e)
                & (rrs["longitude"] > w)
            ),
            drop=True,
        )
    print('north',n,'south',s,'east',e,'west',w)
    

    with xr.open_dataset(rrs_path) as ds:
        time_coverage_start = ds.attrs['time_coverage_start']
        month = datetime.strptime(time_coverage_start,'%Y-%m-%dT%H:%M:%S.%fZ').month
    month = str(month)

    sss_key = 'sss' + month
    sst_key = 'data' + month

    with xr.open_dataset(sal_path) as ds:
        if 'smap_sss' in ds.variables:
            # use SMAP salinity
            sal = xr.open_dataset(sal_path)['smap_sss']
            sal = sal.interp(longitude=rrs_box.longitude, latitude=rrs_box.latitude, method='nearest')
        elif sss_key in ds.variables:
            # use climatology
            sal = xr.open_dataset(sal_path)
            sal[sss_key] = sal[sss_key].assign_coords({
                'Number of Latitudes': sal['Latitude'],
                'Number of Longitudes': sal['Longitude']
            })

            sal = sal.rename({
                'Number of Latitudes': 'lat',
                'Number of Longitudes': 'lon'
            })

            # re-align longitude coords to -180 to 180 
            sal = sal.assign_coords({
                "lon": (((sal.lon + 180) % 360) - 180)
            })

            sal = sal.sortby('lon')
            sal = sal[sss_key]
            sal = sal.interp(lon=rrs_box.longitude, lat=rrs_box.latitude, method='nearest')

    with xr.open_dataset(temp_path) as ds:
        if 'analysed_sst' in ds.variables:
            temp = xr.open_dataset(temp_path)['analysed_sst']
            temp = temp.interp(lon=rrs_box.longitude, lat=rrs_box.latitude, method='nearest').squeeze()
            temp = temp - 273.15
        elif sst_key in ds:
            # use climatology
            temp = xr.open_dataset(temp_path)
            temp_lat_dim = 2 * (int(month)-1)
            temp_lon_dim = temp_lat_dim + 1

            if month == '12':
                dim1 = 'Latitude'
                dim2 = 'Longitude'
            else:
                dim1 = 'fakeDim' + str(temp_lat_dim)
                dim2 = 'fakeDim' + str(temp_lon_dim)
            temp = temp.rename({dim1: 'Latitude', dim2: 'Longitude'})

            temp = temp[sst_key]
            temp = temp.interp(Longitude=rrs_box.longitude, Latitude=rrs_box.latitude, method='nearest')
            temp = temp.slope * temp + temp.intercept


    rrs_flat = rrs_box.stack(pixel=("number_of_lines", "pixels_per_line"))
    rrs_flat = rrs_flat.transpose("pixel", "wavelength")
    rrs_flat = rrs_flat.interp(wavelength=np.arange(346,720)) 

    rrs_np = rrs_flat.to_numpy().reshape(-1, rrs_flat.wavelength.size) 
    rrs_df = pd.DataFrame(rrs_np, columns=np.arange(346,720,1))

    sal_np = sal.to_numpy().flatten()
    temp_np = temp.to_numpy().flatten()

    return rrs_df, sal_np, temp_np

def plot_pigments(data, lower_bound, upper_bound, title):
    '''
    Plots the pigment data from an L2 file with lat/lon coordinates using a color map

    Paramaters:
    -----------
    data : Xarray data array
        Contains pigment values to be plotted.
    lower_bound : float
        The lowest value represented on the color scale.
    upper_bound : float
        The upper value represented on the color scale.
    '''
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    ax.gridlines(draw_labels={'bottom':'x','left':'y'})
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
    data.plot(
        x='longitude',
        y='latitude',
        vmin=lower_bound,
        vmax=upper_bound,
        ax=ax
    )
    plt.show()

def sdp_from_pace(pace_file, output_str, sss_file=None, sst_file=None, bbox=None):
    '''
    Apply SDP to PACE L2 Rrs data to generate pigment concentrations. Saves the results as a netCDF file.
    
    :param pace_file: File path to PACE L2 AOP data. 
    :param output_str: file name string to save results as.
    :param sss_file: SMAP or climatology salinity file path.
    :param sst_file: GHRSST or climatology temperature file path.
    :param bbox: A tuple representing spatial bounds in the form (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).
        Default is None, which means the entire L2 granule will be processed
    '''

    if sss_file is None:
        sss_file = Path('climatology', 'sss_climatology_woa2009.nc')

    if sst_file is None:
        sst_file = Path('climatology', 'sst_climatology.nc')

    interp_start = perf_counter()
    print('starting interpolation')

    rrs_interp, sss_interp, sst_interp = interpolate_coords(pace_file, sss_file, sst_file, bbox)
    interp_end = perf_counter()
    print('interpolation complete:', interp_end - interp_start)

    print('rrs_interp shape:', rrs_interp.shape)
    wl = np.arange(400,701,1)

    rrs_cutoff = rrs_interp.loc[:,400:700]

    # remove nan values
    nanmask = np.isnan(rrs_cutoff.iloc[:,0].values) | np.isnan(sss_interp) | np.isnan(sst_interp)
    rrs_cutoff = rrs_cutoff[~nanmask]
    sss_interp = sss_interp[~nanmask]
    sst_interp = sst_interp[~nanmask]

    print(sst_interp.shape, sss_interp.shape, rrs_cutoff.shape)
    print(np.isnan(rrs_cutoff.iloc[:,0]).sum())

    sdp = run_sdp(rrs_cutoff, wl, sst_interp, sss_interp)

    print('sdp calculation complete')

    # put nan values back in (for mapping later)
    full_sdp = pd.DataFrame(np.nan, index=np.arange(nanmask.shape[0]), columns=sdp.columns)
    full_sdp.loc[~nanmask, :] = sdp.values
    sdp = full_sdp

    # add lat/long coords
    nav_data = xr.open_dataset(pace_file, group="navigation_data")
    nav_data = nav_data.set_coords(("latitude", "longitude"))
    
    if bbox is not None:
        w,s,e,n = bbox[0],bbox[1],bbox[2],bbox[3]

        try:
            nav_box = nav_data.where(
                (
                    (nav_data["latitude"] > s)
                    & (nav_data["latitude"] < n)
                    & (nav_data["longitude"] < e)
                    & (nav_data["longitude"] > w)
                ),
                drop=True,
            )
        except ValueError:
            print('Boundary box is outside of the granule boundary.')
            raise
    else:
        n = nav_data.latitude.values.max()
        s = nav_data.latitude.values.min() 
        e = nav_data.longitude.values.max()
        w = nav_data.longitude.values.min()

        nav_box = nav_data.where(
            (
                (nav_data["latitude"] > s)
                & (nav_data["latitude"] < n)
                & (nav_data["longitude"] < e)
                & (nav_data["longitude"] > w)
            ),
            drop=True,
        )

    
    number_of_lines = int(nav_box.latitude.number_of_lines.shape[0])
    pixels_per_line = int(nav_box.latitude.pixels_per_line.shape[0])
    
    chla = sdp['Tchla'].values.reshape(number_of_lines, pixels_per_line)
    chlb = sdp['MVchlb'].values.reshape(number_of_lines, pixels_per_line)
    chlc12 = sdp['Chlc12'].values.reshape(number_of_lines, pixels_per_line)
    zea = sdp['Zea'].values.reshape(number_of_lines, pixels_per_line)
    dvchla = sdp['DVchla'].values.reshape(number_of_lines, pixels_per_line)
    butfuco = sdp['ButFuco'].values.reshape(number_of_lines, pixels_per_line)
    hexfuco = sdp['HexFuco'].values.reshape(number_of_lines, pixels_per_line)
    allo = sdp['Allo'].values.reshape(number_of_lines, pixels_per_line)
    neo = sdp['Neo'].values.reshape(number_of_lines, pixels_per_line)
    viola = sdp['Viola'].values.reshape(number_of_lines, pixels_per_line)
    fuco = sdp['Fuco'].values.reshape(number_of_lines, pixels_per_line)
    chlc3 = sdp['Chlc3'].values.reshape(number_of_lines, pixels_per_line)
    perid = sdp['Perid'].values.reshape(number_of_lines, pixels_per_line)
    
    pigments = xr.Dataset(
        {
            'chla': (['number_of_lines', 'pixels_per_line'], chla),
            'chlb': (['number_of_lines', 'pixels_per_line'], chlb),
            'chlc': (['number_of_lines', 'pixels_per_line'], chlc12),
            'zea': (['number_of_lines', 'pixels_per_line'], zea),
            'dvchla':  (['number_of_lines', 'pixels_per_line'], dvchla),
            'butfuco':  (['number_of_lines', 'pixels_per_line'], butfuco),
            'hexfuco':  (['number_of_lines', 'pixels_per_line'], hexfuco),
            'allo':  (['number_of_lines', 'pixels_per_line'], allo),
            'neo':  (['number_of_lines', 'pixels_per_line'], neo),
            'viola':  (['number_of_lines', 'pixels_per_line'], viola),
            'fuco': (['number_of_lines', 'pixels_per_line'], fuco),
            'chlc3':  (['number_of_lines', 'pixels_per_line'], chlc3),
            'perid':  (['number_of_lines', 'pixels_per_line'], perid),
        },
        coords={
            'number_of_lines': np.arange(number_of_lines),
            'pixels_per_line': np.arange(pixels_per_line),
        },
    )

    # add lat/lon coords
    pigments = xr.merge((pigments, nav_box.coords))

    #results_str = 'sdp_pigments-' + output_str

    pigments.to_netcdf(output_str)