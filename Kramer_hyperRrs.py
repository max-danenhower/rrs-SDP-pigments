from typing import Tuple, Union

import numpy as np
from scipy.optimize import fmin
import pandas as pd

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

    # TODO: Support S and Tc as column vectors
    for param in [S, Tc]:
        if type(param) not in (int, float) or isinstance(param, np.ndarray):
            raise NotImplementedError("S and Tc must be scalar.")

    # values of the constants
    Na = 6.0221417930e23  # Avogadro's constant
    Kbz = 1.3806503e-23  # Boltzmann constant
    Tk = Tc + 273.15  # Absolute temperature
    M0 = 18e-3  # Molecular weight of water in kg/mol

    theta = np.linspace(0.0, 180.0, 18_001)

    rad = theta * np.pi/180  # angle in radians as a 1-d array

    # nsw: absolute refractive index of seawater
    # dnds: partial derivative of seawater refractive index w.r.t. salinity
    nsw, dnds = RInw(lambda_, Tc, S)

    # isothermal compressibility is from Lepple & Millero (1971,Deep Sea-Research), pages 10-11
    # The error ~ +/-0.004e-6 bar^-1
    IsoComp = BetaT(Tc, S)

    # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
    density_sw = rho_sw(Tc, S)

    # water activity data of seawater is from Millero and Leung (1976,American
    # Journal of Science,276,1035-1077). Table 19 was reproduced using
    # Eq.(14,22,23,88,107) then were fitted to polynominal equation.
    # dlnawds is partial derivative of natural logarithm of water activity
    # w.r.t.salinity
    dlnawds = dlnasw_ds(Tc, S)

    # density derivative of refractive index from PMH model
    DFRI = PMH(nsw)  # PMH model

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

    for i, value in enumerate(rad):  # TODO: Is there a better way to do this?
        if np.rad2deg(value) >= 124:
            rad124 = i
            break
    
    betasw124 = (
        beta90sw * (1 + ((np.cos(rad[rad124])) ** 2) * (1 - delta) / (1 + delta))
    )

    return betasw124, bsw, beta90sw, theta

def gsm_cost(IOPs, rrs, aw, bbw, bbpstar, A, B, admstar):
    g = np.array([0.0949, 0.0794])  # Constants from Gordon et al., 1988

    aph = A * IOPs[0]**B
    a = aw + aph + (IOPs[1] * admstar)
    bb = bbw + IOPs[2] * bbpstar
    x = bb / (a + bb)
    
    rrspred = (g[0] + g[1] * x) * x
    cost = np.sum((rrs - rrspred)**2)

    return cost

def gsm_invert(rrs, aw, bbw, bbpstar, A, B, admstar, pnb):
    """
    Inputs:
      rrs:       2D array (n_samples x n_bands)
      aw, bbw:   1D arrays (n_bands)
      bbpstar:   1D array (n_bands)
      A, B:      1D arrays (n_bands)
      admstar:   1D array (n_bands)
      pnb:       [unused in this code, kept for signature compatibility]
    Outputs:
      IOPs:      2D array (n_samples x 3)
      output:    last optimization result dict (from fmin)
    """

    n_samples = rrs.shape[0]
    IOPs = np.full((n_samples, 3), np.nan)
    cost = np.full(n_samples, np.nan)
    IOPSinit = [0.15, 0.01, 0.0029]
    output = None  # Store last output

    for i in range(n_samples):
        rrs_obs = rrs[i, :]

        def cost_fn(IOPs_trial):
            return gsm_cost(IOPs_trial, rrs_obs, aw, bbw, bbpstar, A, B, admstar)

        iops_opt, final_cost, _, exitflag, out = fmin(
            cost_fn,
            IOPSinit,
            xtol=1e-9,
            ftol=1e-9,
            maxfun=2000,
            maxiter=2000,
            full_output=True,
            disp=False
        )

        if exitflag == 0:  # 0 means converged in scipy fmin
            IOPs[i, :] = iops_opt
            IOPSinit = iops_opt  # Update initial guess for next run
            output = out

    return IOPs, output

def get_rrs_residuals(Rrs, temp, sal, chl):
    n = len(temp) # number of samples/spectra

    rrs = Rrs / (0.52 + 1.7 * Rrs)
    wave = np.arange(400,701)

    asw = pd.read_csv('aw_mcf16_350_700_1nm.csv', header=0)
    asw = asw.iloc[50:,1]

    AB_coefs = pd.read_csv('aph_A_B_Coeffs_Sasha_RSE_paper.csv', header=0)
    A = AB_coefs.iloc[50:,1]
    B = AB_coefs.iloc[50:,2]

    Rrs_490 = Rrs['490'].values
    Rrs_555 = Rrs['555'].values

    acdm_s = -(0.01447 + 0.00033 * Rrs_490 / Rrs_555)  
    acdm = np.exp(np.outer(acdm_s, wave - 443))  # shape: (n_samples, n_wavelengths)

    # Assuming T and S are iterable (e.g., lists or arrays) and wave is defined
    bsw = []

    for i in range(n):
        _, bsw_i, _ = betasw124_ZHH2009(wave, float(sal[i]), float(temp[i]))  # use None for empty param
        bsw.append(bsw_i)

    bsw = np.array(bsw)         # shape (1, n) if bsw_i is 1D array length n
    bbsw = 0.5 * bsw.T         # transpose to shape (n, 1) and multiply by 0.5

    Rrs_440 = Rrs['440'].values

    bbp_s = 2.0 * (1 - 1.2 * np.exp(-0.9 * Rrs_440 / Rrs_555))
    bbp = (443 / wave.reshape(-1, 1)) ** bbp_s  # shapes must be compatible for broadcasting

    IOPs = np.empty((n, 3))  # adjust 3 if needed

    for i in range(n):  # Python 0-based index
        # Extract vectors and transpose vectors
        rrs_i = rrs.iloc[:, i].values.T         
        asw_t = asw.values.T                     
        bbsw_i = bbsw.iloc[:, i].values.T       
        bbp_i = bbp.iloc[:, i].values.T        
        A_t = A.values.T                     
        B_t = B.values.T                     
        acdm_i = acdm.iloc[:, i].values.T  

        # Call your gsm_invert function
        iops_i, _ = gsm_invert(rrs_i, asw_t, bbsw_i, bbp_i, A_t, B_t, acdm_i)

        # Store result
        IOPs[i, :] = iops_i

    print(IOPs)



def read_inputs():
    data = pd.read_csv('EXP_NP_NA_HPLC_Rrs.csv', header=0)

    sal = data.loc[:,'Salinity']
    temp = data.loc[:,'Temperature']
    Rrs = data.loc[:,'Rrs400':]
    chl = data.loc[:,'Tchla (mg m^-3)':'Pras']

    return Rrs, sal, temp, chl

if __name__ == "__main__":
    Rrs, sal, temp, chl = read_inputs()

    get_rrs_residuals(Rrs, sal, temp, chl)