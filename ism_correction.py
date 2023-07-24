# -*- coding: utf-8 -*-
import sys
import idlsave
import numpy as np
import matplotlib.pyplot as plt
import astropy.modeling.functional_models as am
import astropy.constants as ac
from scipy.ndimage import convolve1d
import scipy.special as ss
import scipy.constants as sc


"""
This code derives the correction to the transit depth
caused by absorption from the interstellar medium.

Execution
---------
To run the code, execute from the terminal:
$ ./ism_correction.py

Notes
-----

Package requirements
--------------------
idlsave
astropy

Written by
----------
A. G. Sreejith     Space Research Institute (IWF), Austria
Patricio Cubillos  IWF, Austria
luca Fossati       IWF, Austria

Reference
---------
Impact of MgII interstellar medium absorption on near-ultraviolet exoplanet
 transit measurementsOn the effect of ISM absorption on stellar activity
 measurements and its relevance for exoplanet studies.
Sreejith et al. (2023)
"""

# Constants:
rsun  = ac.R_sun.cgs.value           # Sun radius,        cm
AU    = ac.au.cgs.value              # Astronomical Unit, cm
vc    = ac.c.to("km/s").value        # Speed of light, km/s
c0    = ac.c.cgs.value               # Speed of light, cm/s
cA    = ac.c.to("angstrom/s").value  # Speed of light, angstrom/s
sigma = ac.sigma_sb.cgs.value        # Stefan-Boltzmann constant, erg/(cm**2 s K**4)
k_B   = ac.k_B.cgs.value             # Boltzmann constant, erg/K = g cm**2/(s**2 K)
N_A   = ac.N_A.value                 # Avagadro constant, mol-1

# Parameters for MgII:

MgII1w      = 2795.5280 # MgII wavelength
MgII1_loggf = 0.085     # MgII loggf 0.100 VALD/NIST
MgII1_stark = -5.680    # Mg II  stark damping constant
sigmaMg21   = 0.288     # Mg II line width

MgII2w      = 2802.7050 # Mg II wavelength
MgII2_loggf = -0.218    # Mg II loggf-0.210 VALD/NIST
MgII2_stark =-5.680     # Mg II stark damping constant
sigmaMg22   = 0.257     # Mg II line width

# Ratio of the two Mg II lines
Mgaratio_loggf2to1 = (10**MgII2_loggf)/(10**MgII1_loggf)
Mgratio = Mgaratio_loggf2to1

# ISM fixed parameters:
ISM_b_Mg2   = 3     # b-parameter for the Ca2 ISM lines in km/
fractionMg2 = 0.825 #(Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
Mg_abn      = -5.33 #(Frisch & Slavin 2003; this is the ISM abundance of Mg)
#vr_ISM=0.0         # Radial velocity of the ISM absorption lines in km/s

# interpolation to get BV:
T_book = [
    46000, 43000, 41500, 40000, 39000, 37300, 36500, 35000, 34500,
    33000, 32500, 32000, 31500, 29000, 26000, 24500, 20600, 18500,
    17000, 16700, 15700, 14500, 14000, 12500, 10700, 10400,  9700,
     9200,  8840,  8550,  8270,  8080,  8000,  7800,  7500,  7440,
     7220,  7030,  6810,  6720,  6640,  6510,  6340,  6240,  6170,
     6060,  6000,  5920,  5880,  5770,  5720,  5680,  5660,  5590,
     5530,  5490,  5340,  5280,  5240,  5170,  5140,  5040,  4990,
     4830,  4700,  4600,  4540,  4410,  4330,  4230,  4190,  4070,
     4000,  3940,  3870,  3800,  3700,  3650,  3550,  3500,  3410,
     3250,  3200,  3100,  3030,  3000,  2850,  2710,  2650,  2600,
     2500,  2440,  2400,  2320]

R_book = [
    12.5,  12.3,  11.9,  11.2,  10.6,  10.0,  9.62,  9.11,  8.75,
    8.33,  8.11,  7.8,   7.53,  6.81,  5.72,  4.89,  3.85,  3.45,
    3.48,  3.41,  3.4,   2.95,  2.99,  2.91,  2.28,  2.26,  2.09,
    2.0,   1.97,  2.01,  1.94,  1.94,  1.93,  1.86,  1.81,  1.84,
    1.79,  1.64,  1.61,  1.6,   1.53,  1.46,  1.36,  1.30,  1.25,
    1.23,  1.18,  1.12,  1.12,  1.01,  1.01,  0.986, 0.982, 0.939,
    0.949, 0.909, 0.876, 0.817, 0.828, 0.814, 0.809, 0.763, 0.742,
    0.729, 0.72,  0.726, 0.737, 0.698, 0.672, 0.661, 0.656, 0.654,
    0.587, 0.552, 0.559, 0.535, 0.496, 0.460, 0.434, 0.393, 0.369,
    0.291, 0.258, 0.243, 0.199, 0.149, 0.127, 0.129, 0.118, 0.112,
    0.111, 0.107, 0.095, 0.104]
Sp_book = [
    'O3V', 'O4V', 'O5V', 'O5.5V', 'O6V', 'O6.5V', 'O7V', 'O7.5V', 'O8V',
    'O8.5V', 'O9V', 'O9.5V', 'B0V', 'B0.5V', 'B1V', 'B1.5V', 'B2V', 'B2.5V',
    'B3V', 'B4V', 'B5V', 'B6V', 'B7V', 'B8V', 'B9V', 'B9.5V', 'A0V',
    'A1V', 'A2V', 'A3V', 'A4V', 'A5V', 'A6V', 'A7V', 'A8V', 'A9V',
    'F0V', 'F1V', 'F2V', 'F3V', 'F4V', 'F5V', 'F6V', 'F7V', 'F8V',
    'F9V', 'F9.5V', 'G0V', 'G1V', 'G2V', 'G3V', 'G4V', 'G5V', 'G6V',
    'G7V', 'G8V', 'G9V', 'K0V', 'K0.5V', 'K1V', 'K1.5V', 'K2V', 'K2.5V',
    'K3V', 'K3.5V', 'K4V', 'K4.5V', 'K5V', 'K5.5V', 'K6V', 'K6.5V', 'K7V',
    'K8V', 'K9V', 'M0V', 'M0.5V', 'M1V', 'M1.5V', 'M2V', 'M2.5V', 'M3V',
    'M3.5V', 'M4V', 'M4.5V', 'M5V', 'M5.5V', 'M6V', 'M6.5V', 'M7V', 'M7.5V',
    'M8V', 'M8.5V', 'M9V', 'M9.5V']

# Load the stellar fluxes:
ll = idlsave.read('LLfluxes.sav', verbose=False).ll

flux = np.zeros((3,np.size(ll[:,0])))
flux[0] = ll[:,0]  # Wavelength (FINDME)?
wavelength = ll[:,0]

class Voigt(object):
    r"""
    1D Voigt profile model.

    Parameters
    ----------
    x0: Float
       Line center location.
    hwhmL: Float
       Half-width at half maximum of the Lorentz distribution.
    hwhmG: Float
       Half-width at half maximum of the Gaussian distribution.
    scale: Float
       Scale of the profile (scale=1 returns a profile with integral=1.0).

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyratbay.opacity.broadening as b
    >>> Nl = 5
    >>> Nw = 10.0
    >>> hG = 1.0
    >>> HL = np.logspace(-2, 2, Nl)
    >>> l = b.Lorentz(x0=0.0)
    >>> d = b.Gauss  (x0=0.0, hwhm=hG)
    >>> v = b.Voigt  (x0=0.0, hwhmG=hG)

    >>> plt.figure(11, (6,6))
    >>> plt.clf()
    >>> plt.subplots_adjust(0.15, 0.1, 0.95, 0.95, wspace=0, hspace=0)
    >>> for i in np.arange(Nl):
    >>>   hL = HL[i]
    >>>   ax = plt.subplot(Nl, 1, 1+i)
    >>>   v.hwhmL = hL
    >>>   l.hwhm  = hL
    >>>   width = 0.5346*hL + np.sqrt(0.2166*hL**2+hG**2)
    >>>   x = np.arange(-Nw*width, Nw*width, width/1000.0)
    >>>   plt.plot(x/width, l(x), lw=2.0, color="b",         label="Lorentz")
    >>>   plt.plot(x/width, d(x), lw=2.0, color="limegreen", label="Doppler")
    >>>   plt.plot(x/width, v(x), lw=2.0, color="orange",    label="Voigt",
    >>>            dashes=(8,2))
    >>>   plt.ylim(np.amin([l(x), v(x)]), 3*np.amax([l(x), v(x), d(x)]))
    >>>   ax.set_yscale("log")
    >>>   plt.text(0.025, 0.75, r"$\rm HW_L/HW_G={:4g}$".format(hL/hG),
    >>>            transform=ax.transAxes)
    >>>   plt.xlim(-Nw, Nw)
    >>>   plt.xlabel(r"$\rm x/HW_V$", fontsize=12)
    >>>   plt.ylabel(r"$\rm Profile$")
    >>>   if i != Nl-1:
    >>>       ax.set_xticklabels([""])
    >>>   if i == 0:
    >>>       plt.legend(loc="upper right", fontsize=11)
    """
    def __init__(self, x0=0.0, hwhmL=1.0, hwhmG=1.0, scale=1.0):
        # Profile parameters:
        self.x0    = x0
        self.hwhmL = hwhmL
        self.hwhmG = hwhmG
        self.scale = scale
        # Constants:
        self._A = np.array([-1.2150, -1.3509, -1.2150, -1.3509])
        self._B = np.array([ 1.2359,  0.3786, -1.2359, -0.3786])
        self._C = np.array([-0.3085,  0.5906, -0.3085,  0.5906])
        self._D = np.array([ 0.0210, -1.1858, -0.0210,  1.1858])
        self._sqrtln2 = np.sqrt(np.log(2.0))
        self._sqrtpi  = np.sqrt(np.pi)


    def __call__(self, x):
        return self.eval(x)


    def eval(self, x):
        """
        Compute Voigt profile over the specified coordinates range.

        Parameters
        ----------
        x: 1D float ndarray
           Input coordinates where to evaluate the profile.

        Returns
        -------
        v: 1D float ndarray
           The line profile at the x locations.
        """
        if self.hwhmL/self.hwhmG < 0.1:
            sigma = self.hwhmG / (self._sqrtln2 * np.sqrt(2))
            z = (x + 1j * self.hwhmL - self.x0) / (sigma * np.sqrt(2))
            return self.scale * ss.wofz(z).real / (sigma * np.sqrt(2*np.pi))

        # This is faster than the previous script (but fails for HWl/HWg > 1.0):
        X = (x-self.x0) * self._sqrtln2 / self.hwhmG
        Y = self.hwhmL * self._sqrtln2 / self.hwhmG

        V = 0.0
        for i in np.arange(4):
            V += (self._C[i]*(Y-self._A[i]) + self._D[i]*(X-self._B[i])) \
                 / ((Y-self._A[i])**2 + (X-self._B[i])**2)
        V /= np.pi * self.hwhmL
        return \
            self.scale * self.hwhmL/self.hwhmG * self._sqrtpi*self._sqrtln2 * V





def gaussian(wavelength, wl0, sigma, scale=1.0):
    """
    Compute a Gaussian function centered at wl0, with width sigma, and
    height scale at wl0.
    """
    return scale * np.exp(-0.5*((wavelength-wl0)/sigma)**2)


def findel(value, array):
    """
    Get the indexof the closest element in array  to value.
    """
    arrays = np.array(array)
    diff = np.abs(arrays - value)
    ind  = np.argmin(diff)
    return ind

#########################################

def wtd_mean(values, weights):
    # Error Check and Parameter Setting
    values  = np.asarray(values,  np.float)
    weights = np.asarray(weights, np.float)

    # Calculation
    good_pts_val = np.isfinite(values)   # find non-NaN values
    if np.sum(good_pts_val) > 0:
        values  = values [good_pts_val]
        weights = weights[good_pts_val]

    good_pts_wts = np.isfinite(weights)  # find also w/ corresp.
    if np.sum(good_pts_wts) > 0:   # non-NaN weights
        values  = values [good_pts_wts]
        weights = weights[good_pts_wts]

    if np.any(weights < 0.0):
        print('invalid weights')
        sys.exit(0)

    return np.average(values, weights=weights)

#########################################

def voigtq(wavelength, absorber, line):
    bnorm = absorber["B"] / vc
    # Doppler width (Hz):
    vd = absorber["B"]*sc.kilo / (line["wave"] * sc.angstrom)

    vel = (wavelength/(line["wave"]*(1.0+absorber["Z"])) - 1.0)  / bnorm
    a = line["gamma"] / (4*np.pi * vd)

    idx_wings = np.abs(vel) >= 10.0
    idx_core = np.abs(vel) < 10.0

    vo = vel*0.0
    if np.any(idx_wings) > 0:
          vel2 = vel[idx_wings]**2
          hh1 = 0.56419/vel2 + 0.846/vel2**2
          hh3 = -0.56/vel2**2
          vo[idx_wings] = a * (hh1 + a**2 * hh3)

    if np.any(idx_core) > 0:
        x0 = 0.0
        hwhm_L = line["gamma"] / np.sqrt(2) / 2
        hwhm_G = vd * np.sqrt(np.log(2))
        voigt = Voigt(x0, hwhm_L, hwhm_G)
        vo[idx_core] = voigt(vel[idx_core]*vd)
        vo[idx_core] /= np.amax(vo[idx_core])

    tau = 0.014971475*(10.0**absorber["N"]) * line["F"] * vo/vd

    return np.exp(-tau)


def find_nearest(array, value):
    """
    Find the nearest value in an array
    """
    array = np.asarray(array)
    idx   = (np.abs(array - value)).argmin()
    return idx


def trapz_error(wavelength,flux, error):
    """
    Trapezoidal integration with error propagation.
    """
    integ = 0.0
    var   = 0.0
    dwl   = np.ediff1d(wavelength, 0, 0)
    #ddwl = dwl[1:] + dwl[:-1]
    ddwl  = wavelength[1:]-wavelength[:-1]
    fluxer= flux[1:]+flux[:-1]

    # Standard trapezoidal integration:
    integ = np.sum( ddwl * fluxer * 0.5)
    #var   = np.sum(0.25 * (ddwl * error)**2)

    return integ#, np.sqrt(var)

def gaussbroad(w,s,hwhm):
    #Smooths a spectrum by convolution with a gaussian of specified hwhm.
    # w (input vector) wavelength scale of spectrum to be smoothed
    # s (input vector) spectrum to be smoothed
    # hwhm (input scalar) half width at half maximum of smoothing gaussian.
    #Returns a vector containing the gaussian-smoothed spectrum.
    #Edit History:
    #  -Dec-90 GB,GM Rewrote with fourier convolution algorithm.
    #  -Jul-91 AL	Translated from ANA to IDL.
    #22-Sep-91 JAV	Relaxed constant dispersion check# vectorized, 50% faster.
    #05-Jul-92 JAV	Converted to function, handle nonpositive hwhm.

    #Warn user if hwhm is negative.
    #  if hwhm lt 0.0 then $
    #    message,/info,'Warning! Forcing negative smoothing width to zero.'
        #
    #Return input argument if half-width is nonpositive.
    if hwhm <= 0.0:
        return(s)			#true: no broadening

    #Calculate (uniform) dispersion.
    dw = (w[-1] - w[0]) / len(w)		#wavelength change per pixel
    #gauus=make
    for i in range(0, len(w)):
        #Make smoothing gaussian# extend to 4 sigma.
        if(hwhm > 5*(w[-1] - w[0])):
            return np.full(len(w),np.sum(s)/len(w))
        nhalf = int(3.3972872*hwhm/dw)		## points in half gaussian
        ng = 2 * nhalf + 1				## points in gaussian (odd!)
        wg = dw * (np.arange(ng) - (ng-1)/2.0)	#wavelength scale of gaussian
        xg = ( (0.83255461) / hwhm) * wg 		#convenient absisca
        gpro = ( (0.46974832) * dw / hwhm) * np.exp(-xg*xg)#unit area w/ FWHM
        gpro=gpro/np.sum(gpro)

    #Pad spectrum ends to minimize impact of Fourier ringing.
    npad = nhalf + 2				## pad pixels on each end
    spad = np.concatenate((np.full(npad,s[0]),s,np.full(npad,s[-1])))
    #Convolve & trim.
    #sout = np.convolve(spad,gpro,mode='valid')#convolve with gaussian
    sout =convolve1d(spad,gpro)
    sout = sout[npad:npad+len(w)]  #trim to original data/length
    return sout	 #return broadened spectrum.


def waveres(wave,spectra,fwhm):
    smoothedflux = gaussbroad(wave,spectra,fwhm/2.0)
    return wave, smoothedflux


def mg2_noism(flux,tds,dwl,fwhm,Mgaratio,MgII2w,sigmaMg22,MgII1w,sigmaMg21,E):
    flux_noism = np.zeros((2,np.size(flux[0])))
    #mg2_noism_flux,flux_return
    Mg21em = E/(1.+Mgaratio)
    Mg22em = Mgaratio*E/(1.+Mgaratio)
    gaussMg22 = gaussian(flux[0],MgII2w,sigmaMg22,0.3989*Mg22em/sigmaMg22)
    gaussMg21 = gaussian(flux[0],MgII1w,sigmaMg21,0.3989*Mg21em/sigmaMg21)

    gaussMg2 = gaussMg21 + gaussMg22
    flux_noism[0] = flux[0]
    flux_noism[1] = flux[1] + gaussMg2
    if dwl <= (MgII2w-MgII1w):
        mg2k_wst = MgII1w-(dwl/2.0)
        mg2k_wen = MgII1w+(dwl/2.0)
        mg2h_wst = MgII2w-(dwl/2.0)
        mg2h_wen = MgII2w+(dwl/2.0)
    else:
        mg2_wst = ((MgII2w+MgII1w)/2.0)-(dwl)
        mg2_wen = ((MgII2w+MgII1w)/2.0)+(dwl)

    flux_noism[1] = flux_noism[1]*tds

    wave_noism = flux[0]
    spectra_noism = flux_noism[1]
    st = findel(wave_noism, 2700)
    en = findel(wave_noism, 2900)
    wave_new = wave_noism[st:en]
    photons_star_new = spectra_noism[st:en]
    hwhm = fwhm/2.0
    #convolution with instrument response
    smoothedflux_noism = gaussbroad(wave_new,photons_star_new,hwhm)
    if dwl <= (MgII2w-MgII1w):
        mg2k_st = find_nearest(wave_new,mg2k_wst )
        mg2k_en = find_nearest(wave_new, mg2k_wen)
        mg2h_st = find_nearest(wave_new, mg2h_wst)
        mg2h_en = find_nearest(wave_new, mg2h_wen)

        mg2k_wave=wave_new[mg2k_st:mg2k_en]
        mg2k_flux=smoothedflux_noism[mg2k_st:mg2k_en]
        mg2h_wave=wave_new[mg2h_st:mg2h_en]
        mg2h_flux=smoothedflux_noism[mg2h_st:mg2h_en]
        mg2k_error=np.zeros(len(mg2k_wave))
        mg2k=trapz_error(mg2k_wave,mg2k_flux,mg2k_error)
        mg2h_error=np.zeros(len(mg2h_wave))
        mg2h=trapz_error(mg2h_wave,mg2h_flux,mg2h_error)
        mg2_noism=mg2k+mg2h
    else:
        mg2_st = find_nearest(wave_new,mg2_wst )
        mg2_en = find_nearest(wave_new, mg2_wen)
        mg2_wave=wave_new[mg2_st:mg2_en]
        mg2_flux=smoothedflux_noism[mg2_st:mg2_en]
        mg2_error=np.zeros(len(mg2_wave))
        mg2hk=trapz_error(mg2_wave,mg2_flux,mg2_error)
        mg2_noism=mg2hk

    return flux_noism, mg2_noism


def mg2_ism(flux_noism, dwl, MgII2w, MgII1w, fwhm, vr_ISM, n_mg2, ISM_b_Mg2):
    if dwl <= (MgII2w-MgII1w):
        mg2k_wst = MgII1w-(dwl/2.0)
        mg2k_wen = MgII1w+(dwl/2.0)
        mg2h_wst = MgII2w-(dwl/2.0)
        mg2h_wen = MgII2w+(dwl/2.0)
    else:
        mg2_wst = ((MgII2w+MgII1w)/2.0)-(dwl)
        mg2_wen = ((MgII2w+MgII1w)/2.0)+(dwl)
    #for MgII doublet
    absorberMg1 = {
        'ion':'MG21',
        'N': n_mg2,
        'B':ISM_b_Mg2,
        'Z': 0.0,
    }
    lineMg1 = {
        'ion':'Mg21',
        'wave':MgII1w+MgII1w*vr_ISM/vc,
        'F':10**MgII1_loggf,
        'gamma':10**MgII1_stark,
    }
    ISMMg21 = voigtq(flux_noism[0], absorberMg1, lineMg1)

    absorberMg2 = {
        'ion':'MG22',
        'N':n_mg2,
        'B':ISM_b_Mg2,
        'Z':0.0,
    }
    lineMg2 = {
        'ion':'Mg22',
        'wave':MgII2w+MgII2w*vr_ISM/vc,
        'F':10**MgII2_loggf,
        'gamma':10**MgII2_stark,
    }
    ISMMg22 = voigtq(flux_noism[0], absorberMg2, lineMg2)

    ISM = ISMMg21*ISMMg22
    flux_absorption = ISM
    spectra=flux_noism[1]*flux_absorption
    wave=flux_noism[0]
    st = findel(wave, 2700)
    en = findel(wave, 2900)
    wave_new = wave[st:en]
    photons_star_new  = spectra[st:en]
    hwhm = fwhm/2.0
    smoothedflux = gaussbroad(wave_new,photons_star_new,hwhm)

    if dwl<= (MgII2w-MgII1w):
        mg2k_st = find_nearest(wave_new,mg2k_wst )
        mg2k_en = find_nearest(wave_new, mg2k_wen)
        mg2h_st = find_nearest(wave_new, mg2h_wst)
        mg2h_en = find_nearest(wave_new, mg2h_wen)
        mg2k_wave=wave_new[mg2k_st:mg2k_en]
        mg2k_flux=smoothedflux[mg2k_st:mg2k_en]
        mg2h_wave=wave_new[mg2h_st:mg2h_en]
        mg2h_flux=smoothedflux[mg2h_st:mg2h_en]
        mg2k_error=np.zeros(len(mg2k_wave))
        mg2k=trapz_error(mg2k_wave,mg2k_flux,mg2k_error)
        mg2h_error=np.zeros(len(mg2h_wave))
        mg2h=trapz_error(mg2h_wave,mg2h_flux,mg2h_error)
        mg2=mg2k+mg2h
    else:
        mg2_st = find_nearest(wave_new,mg2_wst )
        mg2_en = find_nearest(wave_new, mg2_wen)
        mg2_wave=wave_new[mg2_st:mg2_en]
        mg2_flux=smoothedflux[mg2_st:mg2_en]
        mg2_error=np.zeros(len(mg2_wave))
        mg2hk=trapz_error(mg2_wave,mg2_flux,mg2_error)
        mg2=mg2hk

    return mg2

def read(msg, default=-9.9):
    """
    Read value from prompt. Return float.
    """
    val = input(msg)
    if val == "":
        return default
    return np.float(val)

def main():
    """
    See docstring at the top of this file.
    """
    # Parse argv:
    if len(sys.argv) > 1:
        inputpara = True
    else:
        inputpara = False

    # User's parameters
    if not inputpara:
        print('Stellar parameters ')
        Teff = read('Enter stellar Teff (K) [valid range (K): 3500--10000]:')
        if Teff < 3495 or Teff > 10000: # Out of bounds:
            print('Invalid temperature, please set at least one valid Teff.')
            return
        T = np.arange(3500, 10100, 100, np.int)
        loc = findel(Teff,T_book)
        radius = read('Enter stellar radius (Rsun)')
        if radius < 0:
            radius = R_book[loc]
        Rmg = read("Enter MgII line core emission at 1AU:")
        if Rmg < 0.0:
            logR = read("Enter logR'HK value (default: -5.1):\n", -5.1)
            if logR < -5.1:
                logR = -5.1
                print(
                    "Warning: the input logR'HK value is smaller than the "
                    "basal level of -5.1.\n"
                    "The logR'HK value is now set to -5.1.")
            if logR < -7.0:
              print("Please enter MgII line core emission at 1AU or logR'HK!")
              return
        
        mgksigma = read(' Please enter the width of  MgII k line (FWHM) [A] '
                        '(if unknown enter -9.9) : ')
        mghsigma = read(' Please enter the width of  MgII h line (FWHM) [A] '
                        '(if unknown enter -9.9) : ')
        print("Planetary parameters.")
        bb_td = read('Enter the broad band transit depth [%]:')
        mg_td = read('Enter the MgII transit depth [%]:')
        p_width = read('Enter the width of planetary absorption feature [A]: ')
        print("ISM parameters (either reddening or ISM column density).")
        red = read(
            'Enter the reddening E(B-V) (mag) (leave blank to set '
            'ISM column density):')
        if red >= 0:
            nh = red*5.8e21
            nmg2 = np.log10(nh*fractionMg2*10.0**Mg_abn)
        if red <= 0.0:
            nmg2 = read('Enter the CaII ISM column density (log10 cm^-2):')
            if nmg2 == -9.9:
                print("Please enter either the reddening or column density.")
                return

        vr_ISM = read(
            'Enter the ISM radial velocity (relative to the stellar radial'
            ' velocity) (km/s) (default=0.0):', 0.0)

        print("Instrument parameters")
        fwhm = read('Enter the resolution of the instrument [A]:')
        dwl = read('Enter the wavelength integration bin [A]:')
        dwl = dwl/2.0
        if p_width <= 0.0:
            p_width = 1.5
        if mgksigma > 0:
            sigmaMg21 = mgksigma*2*np.sqrt(2*np.log(2))
        else:
            sigmaMg21   = 0.288    
        if mghsigma > 0:
            sigmaMg22 = mghsigma*2*np.sqrt(2*np.log(2))
        else:
            sigmaMg22   = 0.257  
        loct = findel(Teff,T)
        mm = loct*2 + 1

        stype = Sp_book[loc]
        flux[1] = ll[:,mm]
        flux[2] = ll[:,mm+1]
        flux[1] = (flux[1]*cA) / (wavelength**2.0)
        flux[2] = (flux[2]*cA) / (wavelength**2.0)
        flux[1] = flux[1] * 4.0*np.pi*(radius*rsun)**2 * 4.0*np.pi
        flux[2] = flux[2] * 4.0*np.pi*(radius*rsun)**2 * 4.0*np.pi

        if Rmg == '-9.9':
            if (stype == 'F5V' or stype == 'F6V' or stype == 'F7V'
                or stype == 'F8V' or stype == 'G0V' or stype == 'G1V'
                or stype == 'G2V' or stype == 'G5V' or stype == 'G8V'):
                c1 = 0.87
                c2 = 5.73
                Rmg = 10**(c1*logR+c2)
            elif (stype == 'K5V' or stype == 'K4V' or stype == 'K3V' or
                  stype == 'K2V' or stype == 'K1V' or stype == 'K0V') :
                c1 = 1.01
                c2 = 6.00
                Rmg = 10**(c1*logR+c2)
            elif (stype == 'M5V' or stype == 'M4V' or stype == 'M3V' or
                  stype == 'M2V' or stype == 'M1V' or stype == 'M0V') :
                c1 = 1.59
                c2 = 6.96
                Rmg = 10**(c1*logR+c2)
            else:
                Rmg = 0.0

        E = Rmg * AU**2
        WL = wavelength
        Rp = np.zeros(len(WL))
        Rp = Rp+bb_td
        lFWHM = p_width  # full width half maximum
        sigma = lFWHM / (2. * np.sqrt(2.* np.log(2.)))
        Rp_Mg1 = mg_td  # peak in planetary radii
        Rp_Mg2 =Rp_Mg1*Mgaratio_loggf2to1

        Rp_Mg21 = Rp_Mg1-bb_td
        Rp_Mg22 = Rp_Mg2-bb_td
        Mg21 = gaussian(WL,MgII1w,sigma, Rp_Mg21)
        Mg22 = gaussian(WL,MgII2w,sigma,Rp_Mg22)
        R = Rp+Mg21+Mg22
        td = np.zeros((2,len(WL)))
        td[1,:] = R
        mg2_noism_flux = np.zeros(2)
        mg2_ism_flux = np.zeros(2)
        tds_noism = np.zeros(2)
        tds_ism = np.zeros(2)
        flux_c = flux
        for k in range(2):
            tdval=(1-(td[k,:]/100.0))
            flux_return, mg2_noism_val = mg2_noism(
                flux_c, tdval, dwl, fwhm, Mgratio,
                MgII2w, sigmaMg22, MgII1w, sigmaMg21, E)
            mg2_noism_flux[k]=mg2_noism_val
            mg2_ism_flux[k] = mg2_ism(
                flux_return, dwl, MgII2w, MgII1w, fwhm, vr_ISM, nmg2, ISM_b_Mg2)
            tds_noism[k]=1.-(mg2_noism_flux[k]/mg2_noism_flux[0])
            tds_ism[k]=1.-(mg2_ism_flux[k]/mg2_ism_flux[0])
        tds_diff=tds_ism[1]-tds_noism[1]

        print('Difference in transit depth [%]',tds_diff*100)
            # Read from input file:

        #log creation
        logfile = open('ism_correction_log','w')
        log ='Mg II ISM correction parameters'
        log = log+'Stellar temperature (K)  : '+str(Teff)+'\n'
        log = log+'Stellar radius (Rsun)    : '+str(radius)+'\n'
        log = log+'Spectral type            : '+stype+'\n'
        try:
            log = log+'Stellar activity index   : '+str(logR)+'\n'
        except:
            pass
        log = log+'Mg II line emission @ 1AU: '+str(Rmg)+'\n'
        log = log+'Planetary Parameters \n'
        log = log+'Broad band transit depth [%]  : '+str(bb_td)+'\n'
        log = log+'Peak MgII k transit depth [%] : '+str(mg_td)+'\n'
        log = log+'Planetary absorption width [A]: '+str(lFWHM)+'\n'
        log = log+'ISM parameters \n'
        try:
            log = log+'reddening E(B-V) [mag]     : '+str(red)+'\n'
        except:
            pass
        log = log+'MgII column density        : '+str(nmg2)+'\n'
        log = log+'ISM b-parameter [km/s]     : '+str(ISM_b_Mg2)+'\n'
        log = log+'ISM radial velocity [km/s] : '+str(vr_ISM)+'\n'
        log = log+'Instrument paramters \n'
        log = log+'Resolution [A]      : '+str(fwhm)+'\n'
        log = log+'Integration bin [A] : '+str(dwl*2)

        logfile.write(log)

'''
# Read from input parameters:
#  elif inputpara:
To be implemented
'''
      

if __name__ == "__main__":
  main()
