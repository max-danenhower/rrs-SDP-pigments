from typing import Tuple, Union

import numpy as np
from scipy.optimize import fmin
import pandas as pd
import sys
import time
import ray

def RInw(
    lambda_: Union[int, float, np.ndarray],
    Tc: Union[int, float],
    S: Union[int, float],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Refractive index of air is from Ciddor (1996,Applied Optics)
    Refractive index of seawater is from Quan and Fry (1994, Applied Optics)
    """

    n_air = (
        1.0 + (5792105.0 / (238.0185 - 1 / (lambda_ / 1e3) ** 2)
        + 167917.0 / (57.362 - 1 / (lambda_ / 1e3) ** 2)) / 1e8
    )
    n0 = 1.31405
    n1 = 1.779e-4
    n2 = -1.05e-6
    n3 = 1.6e-8
    n4 = -2.02e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382
    n9 = 1.1455e6
    
    nsw = (
        n0 + (n1 + n2 * Tc + n3 * Tc ** 2) * S + n4 * Tc ** 2 
        + (n5 + n6 * S + n7 * Tc) / lambda_ + n8 / lambda_ ** 2 + n9
        / lambda_ ** 3
    )
    

    nsw = nsw * n_air
    dnswds = (n1 + n2 * Tc + n3 * Tc ** 2 + n6 / lambda_) * n_air

    return nsw, dnswds


def BetaT(Tc: Union[int, float], S: Union[int, float]) -> float:
    """Pure water secant bulk Millero (1980, Deep-sea Research).
    Isothermal compressibility from Kell sound measurement in pure water.
    Calculate seawater isothermal compressibility from the secant bulk.
    """
    kw = (
        19652.21 + 148.4206 * Tc - 2.327105 * Tc ** 2 + 1.360477e-2 
        * Tc ** 3 - 5.155288e-5 * Tc ** 4
    )
    Btw_cal = 1 / kw
    a0 = 54.6746 - 0.603459 * Tc + 1.09987e-2 * Tc ** 2-6.167e-5 * Tc ** 3
    b0 = 7.944e-2 + 1.6483e-2 * Tc - 5.3009e-4 * Tc ** 2

    Ks = kw + a0 * S + b0 * S ** 1.5
    IsoComp = 1 / Ks * 1e-5

    return IsoComp


def rho_sw(Tc: Union[int, float], S: Union[int, float]) -> float:
    """Density of water and seawater, unit is Kg/m^3, from UNESCO,38,1981.
    TODO: Compare to GSW Oceanographic Toolbox code (or other updated) density equations.
    """
    a0 = 8.24493e-1
    a1 = -4.0899e-3
    a2 = 7.6438e-5
    a3 = -8.2467e-7
    a4 = 5.3875e-9
    a5 = -5.72466e-3
    a6 = 1.0227e-4
    a7 = -1.6546e-6
    a8 = 4.8314e-4
    b0 = 999.842594
    b1 = 6.793952e-2
    b2 = -9.09529e-3
    b3 = 1.001685e-4
    b4 = -1.120083e-6
    b5 = 6.536332e-9

    # density for pure water
    density_w = b0 + b1 * Tc + b2 * Tc ** 2 + b3 * Tc ** 3 + b4 * Tc ** 4 + b5 * Tc ** 5
    # density for pure seawater
    density_sw = (
        density_w + ((a0 + a1 * Tc + a2 * Tc ** 2 + a3 * Tc ** 3 + a4 * Tc ** 4) * S
        + (a5 + a6 * Tc + a7 * Tc ** 2) * S ** 1.5 + a8 * S ** 2)
    )
    return density_sw


def dlnasw_ds(Tc: Union[int, float], S: Union[int, float]) -> float:
    """Water activity data of seawater is from Millero and Leung (1976,American
    Journal of Science,276,1035-1077). Table 19 was reproduced using
    Eqs.(14,22,23,88,107) then were fitted to polynominal equation.
    dlnawds is partial derivative of natural logarithm of water activity
    w.r.t.salinity.

    lnaw =  (-1.64555e-6-1.34779e-7*Tc+1.85392e-9*Tc.^2-1.40702e-11*Tc.^3)+......
            (-5.58651e-4+2.40452e-7*Tc-3.12165e-9*Tc.^2+2.40808e-11*Tc.^3).*S+......
            (1.79613e-5-9.9422e-8*Tc+2.08919e-9*Tc.^2-1.39872e-11*Tc.^3).*S.^1.5+......
            (-2.31065e-6-1.37674e-9*Tc-1.93316e-11*Tc.^2).*S.^2;
    
    density derivative of refractive index from PMH model
    """

    dlnawds = (
        (-5.58651e-4 + 2.40452e-7 * Tc - 3.12165e-9 * Tc ** 2 + 2.40808e-11 * Tc ** 3)
        + 1.5 * (1.79613e-5 - 9.9422e-8 * Tc + 2.08919e-9 * Tc ** 2 - 1.39872e-11 * Tc ** 3) * S ** 0.5
        + 2 * (-2.31065e-6 - 1.37674e-9 * Tc - 1.93316e-11 * Tc ** 2) * S
    )
    return dlnawds


def PMH(n_wat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    n_wat2 = n_wat ** 2
    n_density_derivative = (
        (n_wat2 - 1) * (1 + 2 / 3 * (n_wat2 + 2)
        * (n_wat / 3 - 1 / 3 / n_wat) ** 2)
    )
    return n_density_derivative

def betasw124_ZHH2009(lambda_, S, Tc, delta=0.039):
    """Scattering by pure seawater: Effect of salinity
    Xiaodong Zhang, Lianbo Hu, and Ming-Xia He, Optics Express, 2009, accepted
    lambda (nm): wavelength
    Tc: temperauter in degree Celsius, must be a scalar
    S: salinity, must be scalar
    delta: depolarization ratio, if not provided, default = 0.039 will be used (from Farinato and Roswell (1976))
    betasw: volume scattering at angles defined by theta. Its size is [x y],
    where x is the number of angles (x = length(theta)) and y is the number
    of wavelengths in lambda (y = length(lambda))
    beta90sw: volume scattering at 90 degree. Its size is [1 y]
    bw: total scattering coefficient. Its size is [1 y]
    for backscattering coefficients, divide total scattering by 2
    Xiaodong Zhang, March 10, 2009

    MODIFIED on 17/05/2011 to be able to process bbp profiles with coincident T and sal profiles
    MODIFIED on 05 Apr 2013 to use 124 degs instead of 117 degs
    """

    # values of the constants
    Na = 6.0221417930e23  # Avogadro's constant
    Kbz = 1.3806503e-23  # Boltzmann constant
    Tk = Tc + 273.15  # Absolute temperature
    M0 = 18e-3  # Molecular weight of water in kg/mol

    theta = np.linspace(0.0, 180.0, 18_001)

    rad = theta * np.pi/180  # angle in radians as a 1-d array

    # nsw: absolute refractive index of seawater
    # dnds: partial derivative of seawater refractive index w.r.t. salinity
    nsw, dnds = RInw(lambda_, Tc, S) #shape (n,wavelengths),(n,wavelengths)

    # isothermal compressibility is from Lepple & Millero (1971,Deep Sea-Research), pages 10-11
    # The error ~ +/-0.004e-6 bar^-1
    IsoComp = BetaT(Tc, S) # shape (n,1)

    # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
    density_sw = rho_sw(Tc, S) # shape (n,1)

    # water activity data of seawater is from Millero and Leung (1976,American
    # Journal of Science,276,1035-1077). Table 19 was reproduced using
    # Eq.(14,22,23,88,107) then were fitted to polynominal equation.
    # dlnawds is partial derivative of natural logarithm of water activity
    # w.r.t.salinity
    dlnawds = dlnasw_ds(Tc, S)

    # density derivative of refractive index from PMH model
    DFRI = PMH(nsw)  # PMH model shape (n,wavelengths)

    # volume scattering at 90 degree due to the density fluctuation
    beta_df = (
        np.pi * np.pi / 2 * ((lambda_ * 1e-9) ** (-4)) * Kbz * Tk * IsoComp * DFRI ** 2
        * (6 + 6 * delta) / (6 - 7 * delta)
    )

    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = S * M0 * dnds ** 2 / density_sw / (-dlnawds) / Na
    beta_cf = (
        2 * np.pi * np.pi * ((lambda_ * 1e-9) ** (-4)) * nsw ** 2
        * (flu_con) * (6 + 6 * delta)/(6 - 7 * delta)
    )

    # total volume scattering at 90 degree
    beta90sw = beta_df + beta_cf
    bsw = 8 * np.pi/3 * beta90sw * (2 + delta) / (1 + delta)

    return None, bsw, beta90sw, theta

def gsm_cost(IOPs, rrs, aw, bbw, bbpstar, A, B, admstar):
    g = np.array([0.0949, 0.0794])  # Constants from Gordon et al., 1988

    aph = A * IOPs[0]**B
    a = aw + aph + (IOPs[1] * admstar)
    bb = bbw + IOPs[2] * bbpstar
    x = bb / (a + bb)
    
    rrspred = (g[0] + g[1] * x) * x
    cost = np.sum((rrs - rrspred)**2)

    return cost

def gsm_invert(rrs, aw, bbw, bbpstar, A, B, admstar):

    n_samples = rrs.shape[0]
    IOPs = np.array([np.nan, np.nan, np.nan])
    cost = np.full(n_samples, np.nan)
    IOPSinit = [0.15, 0.01, 0.0029]
    output = None  # Store last output

    def cost_fn(IOPs_trial):
        return gsm_cost(IOPs_trial, rrs, aw, bbw, bbpstar, A, B, admstar)

    iops_opt, _, _, _, _ = fmin(
        cost_fn,
        IOPSinit,
        xtol=1e-6,
        ftol=1e-6,
        maxfun=2000,
        maxiter=2000,
        full_output=True,
        disp=False
    )

    return iops_opt

@ray.remote(num_cpus=1)
def run_batch(rrs, asw, bbsw, bbp, A, B, acdm):
    IOPs = []
    for i in range(len(rrs)):
        rrs_i = rrs[i, :]
        asw_t = asw               
        bbsw_i = bbsw[:, i]
        bbp_i = bbp[:, i]
        A_t = A
        B_t = B
        acdm_i = acdm[i, :]

        iops_i = gsm_invert(rrs_i, asw_t, bbsw_i, bbp_i, A_t, B_t, acdm_i)
        IOPs.append(iops_i)
    return IOPs

def get_rrs_residuals(Rrs, temp, sal, wavelengths):
    '''
    Make a generic model Rrs spectra and compute Rrs residuals (measured Rrs - modeled Rrs) 
    from Kramer et al. (2022)

    Parameters:
    -----------
    Rrs: pandas DataFrame (n_samples, n_wavelengths)
        Rrs spectra
    temp: numpy array (n_samples)
        temperature values corresponding with Rrs spectra
    sal: numpy array (n_samples)
        salinity values corresponding with Rrs spectra 
    wavelengths: numpy array  (n_wavlengths)
        wavelengths corresponding with Rrs spectra

    Returns:
    --------
    rrsD: pandas DataFrame (n_wavelengths, n_samples)
        just-below surface remote sensing reflectance residual
    RrsD: pandas DataFrame (n_wavelengths, n_samples)
        above surface remote-sensing reflectance residual
    '''
    
    # The model uses reflectance = f(IOPs) from Gordon et al. (1988) which uses
    # below the surface reflectance (rrs = Lu(0-)/Ed(0-)). If you are using 
    # above surface reflectance (Rrs = Lu(0+)/Ed(0+)), you will need to convert
    # your reflectances before running this model, using the equation below from
    # Lee et al. (2002):
    rrs = Rrs / (0.52 + 1.7 * Rrs)

    # define total absorption as a sum of seawater absorption (asw), phytoplankton absorption (aph) 
    # and CDOM plus other detrital matter (acdm)
    asw_chart = pd.read_csv('aw_mcf16_350_700_1nm.csv', header=0)
    AB_coefs = pd.read_csv('aph_A_B_Coeffs_Sasha_RSE_paper.csv', header=0)

    A = np.zeros(len(wavelengths)) 
    B = np.zeros(len(wavelengths)) 
    asw = np.zeros(len(wavelengths)) 


    # aph = import A & B coefficients from aph_A_B_Coeffs_Sasha_RSE_paper.csv
    # aph = A.*chl.^B; inversion model will solve for chl as an output
    for i,w in enumerate(wavelengths): 
        idx = np.argmin(np.abs(w-AB_coefs.iloc[:,0])) 
        A[i] = float(AB_coefs.iloc[idx,1]) 
        B[i] = float(AB_coefs.iloc[idx,2]) 
        asw[i] = float(asw_chart.iloc[idx,1]) 

    # get indices closest to wavelengths 
    i490 = np.argmin(np.abs(wavelengths-490)) 
    i555 = np.argmin(np.abs(wavelengths-555)) 
    i440 = np.argmin(np.abs(wavelengths-440)) 

    # acdm slope is a function of Rrs (just above surface):
    # You will need to define Rrs490 and Rrs555 based on your Rrs data
    Rrs_490 = Rrs.iloc[:,i490].values.astype(np.float64) 
    Rrs_555 = Rrs.iloc[:,i555].values.astype(np.float64) 

    acdm_s = -(0.01447 + 0.00033 * Rrs_490 / Rrs_555)  
    acdm = np.exp(np.outer(acdm_s, wavelengths - 443))  # shape: (n_samples, n_wavelengths)

    # define backscattering as a sum of seawater backscattering (bbsw) and backscattering by particles (bbp)
    # bb_tot = bbsw + bbp
    temp_ = np.asarray(temp)[:,None]
    sal_ = np.asarray(sal)[:,None]
    lambda_ = np.asarray(wavelengths)[None,:]
    
    _,bsw,_,_ = betasw124_ZHH2009(lambda_, sal_, temp_)

    bsw = np.array(bsw)        
    bbsw = 0.5 * bsw.T         

    # bbp slope is a function of rrs (just below surface):
    # You will need to define rrs440 and rrs555 based on your rrs data
    Rrs_440 = Rrs.iloc[:,i440].values.astype(np.float64)
    bbp_s = 2.0 * (1 - 1.2 * np.exp(-0.9 * Rrs_440 / Rrs_555))
    bbp = (443 / wavelengths.reshape(-1, 1)) ** bbp_s

    Rrs_np = rrs.values

    batch_size = min(10_000,int(len(temp)/32))

    batches = [
        (
            Rrs_np[i:i+batch_size,:],
            asw,
            bbsw[:,i:i+batch_size],
            bbp[:,i:i+batch_size],
            A,B,
            acdm[i:i+batch_size,:]
        )
        for i in range(0, len(Rrs_np), batch_size)
    ]

    # to run serially, comment out from here ...
    ray.init(include_dashboard=True)

    print('ray availble resources', ray.available_resources(),'\n')

    # Launch Ray tasks. Run IOP inversion in parallel batches
    futures = [run_batch.remote(*b) for b in batches]
    results = ray.get(futures)  # list of lists, flatten if needed
    IOPs = [res for batch in results for res in batch]
    IOPs = np.array(IOPs)

    ray.shutdown()
    # ... to here

    # Run IOPs inversion serially. Uncomment below. 

    '''
    IOPs = np.empty((len(temp_), 3))

    for i in range(len(temp_)):
        rrs_i = rrs.iloc[i, :].values
        asw_t = asw               
        bbsw_i = bbsw[:, i]
        bbp_i = bbp[:, i]
        A_t = A
        B_t = B
        acdm_i = acdm[i, :]

        if np.isnan(rrs_i).any():
            print('rrs nan',i)
        elif np.isnan(bbsw_i).any():
            print('bbsw nan',i)
        elif np.isnan(bbp_i).any():
            print('bbp nan',i)
        elif np.isnan(acdm_i).any():
            print('acdm nan',i)
        elif np.isnan(asw_t).any():
            print('asw nan',i)
        elif np.isnan(A_t).any():
            print('A nan',i)
        elif np.isnan(B_t).any():
            print('B nan',i)
    
        iops_i = gsm_invert(rrs_i, asw_t, bbsw_i, bbp_i, A_t, B_t, acdm_i)
        IOPs[i, :] = iops_i
    '''


    asw_ = asw[:, np.newaxis]
    A_ = A[:, np.newaxis]
    B_ = B[:, np.newaxis]

    # Reconstruct Rrs for each spectrum
    a = asw_ + (A_ * (IOPs[:, 0]**B_)) + (acdm.T * IOPs[:, 1])
    bb = bbsw + bbp * IOPs[:, 2]

    rrsP = bb / (a + bb)

    # Gordon coefficients
    g1, g2 = 0.0949, 0.0794

    modrrs = (g1 + g2 * rrsP) * rrsP

    # convert back to Rrs
    modRrs = (0.52 * modrrs) / (1 - 1.7 * modrrs)

    # Residual between measured and modeled (to use for Kramer_Rrs_pigments)
    rrsD = rrs.T - modrrs
    RrsD = Rrs.T - modRrs

    return rrsD, RrsD
