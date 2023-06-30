import bz2
import warnings
from pathlib import Path

import pandas as pd
from astropy import units as u
from astropy import constants as const
from astropy.io import fits, ascii
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.modeling.fitting import LinearLSQFitter
from scipy.interpolate import interp1d
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
import numpy as np
from eniric.resample import log_resample
from eniric.broaden import convolution
import matplotlib.pyplot as plt

#AMESCOND_PATH = '/home/vandal/Documents/bpic/analysis/data/external/mass_lum_models/COND03_isochrones_WISE.ascii'
#BCAH15_PATH = '/home/vandal/Documents/bpic/analysis/data/external/mass_lum_models/BCAH15_isochrones.ascii'


def get_wl_range(wl, fl, bounds=(-np.inf, np.inf)):
    if bounds is None:
        return wl, fl
    ind = (wl >= (bounds[0])) & (wl <= (bounds[1]))
    return wl[ind], fl[ind]


def resample_spec(old_wl, old_flux, sampling, R):
    new_wl = log_resample(old_wl, sampling, R)
    new_flux = np.interp(new_wl, old_wl, old_flux)

    return new_wl, new_flux


def shift_wl(wl, dv):

    if dv == 0:
        return wl

    c = const.c.to("km/s").value

    return wl * (1.0 + dv / c)


def shift_spectrum(wl, fl, dv):

    if dv == 0:
        return wl.copy(), fl.copy()

    wls = shift_wl(wl, dv)

    fi = interp1d(wls, fl, bounds_error=False)
    fls = fi(wl)

    # Check where interpolation was invalid
    ind = ~np.isnan(fls)
    wl = wl.copy()[ind]
    fls = fls[ind]

    return wl, fls


def load_phoenix(teff, logg, z, alpha, model_dir, phoenix_wav_path=None, wl_range=None, norm=False):


    teff = np.round(teff, -2)

    leading_zeros = "0" if teff < 1e4 else ""

    logg = np.round(logg * 2) / 2

    za_str = f"{z:.1f}"
    if alpha != 0.0:
        za_str += f".Alpha={alpha:+.2f}"
    z_dir = Path("Z-"+za_str)
    phoenix_file = f"lte{leading_zeros}{teff}-{logg:.2f}-{za_str}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    phoenix_path = model_dir / z_dir / phoenix_file

    # Load the model spectra
    if phoenix_wav_path is None:
        phoenix_wav_path = model_dir.parent / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    wl = fits.getdata(phoenix_wav_path)  # in Angstrom
    wl = wl * 1e-4
    try:
        fl = fits.getdata(phoenix_path)  # erg/s/cm^2/cm
    except:
        teff = np.round(teff / 2, -2) * 2
        phoenix_file = f"lte{leading_zeros}{teff:.0f}-{logg:.2f}-{za_str}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        phoenix_path = model_dir / z_dir / phoenix_file
        fl = fits.getdata(phoenix_path)  # erg/s/cm^2/cm
    fl = (fl * u.erg / u.cm ** 2 / u.s / u.cm).to("W / (m2  um)").value

    if norm:
        fl = fl / np.trapz(fl, wl)

    wl, fl = get_wl_range(wl, fl, wl_range)

    return wl, fl

def load_cifist(teff, logg, z, alpha, model_dir, wl_range=None, norm=False):

    teff = np.round(teff, -2)

    logg = np.round(logg * 2) / 2

    leading_zeros = "0" if teff >= 1000 else "00"
    btsettl_file = f"lte{leading_zeros}{teff/100:.1f}-{logg}-{z}a{alpha:+.1f}.BT-Settl.spec.fits.gz"
    btsettl_path = model_dir / btsettl_file

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Invalid keyword")
        data = fits.getdata(btsettl_path)

    wl = data["Wavelength"].astype(float)  # um
    fl = data["Flux"].astype(float)  # W / (m2 um)

    if norm:
        fl = fl / np.trapz(fl, wl)

    wl, fl = get_wl_range(wl, fl, wl_range)

    return wl, fl


def load_btsettl(teff, logg, z, alpha, model_dir, wl_range=None, norm=False):

    leading_zeros = "0" if teff >= 1000 else "00"
    tstr = teff / 100
    if tstr.is_integer():
        tstr = int(tstr)
    testfile = model_dir / f"lte{leading_zeros}{tstr}-{logg:.1f}-{z}a{alpha:+.1f}.BT-Settl.spec.7.bz2"
    with bz2.BZ2File(testfile) as file:
        lines = file.readlines()

    strlines = [line.decode("utf-8") for line in lines]
    data = ascii.read(
            strlines,
            col_starts=(0, 13),
            col_ends=(12, 25),
            Reader=ascii.FixedWidthNoHeader,
        )
    wl = data["col1"]
    fl_str = data["col2"]
    fl = idl_float(fl_str)
    fl = 10 ** (fl - 8.0)  # now in ergs/cm^2/s/A
    fl = (fl * u.erg / u.cm ** 2 / u.s / u.Angstrom).to("W / (m2  um)").value
    wl = wl * 1e-4
    wl, ind = np.unique(wl, return_index=True)
    fl = fl[ind]

    if norm:
        fl = fl / np.trapz(fl, wl)

    wl, fl = get_wl_range(wl, fl, bounds=wl_range)

    return wl, fl


@np.vectorize
def idl_float(idl_num: str) -> float:
    """
    Convert an IDL string number in scientific notation to a float

    Parameters
    ----------
    idl_num : str
        Input str

    Returns
    -------
    float
        Output float

    Examples
    --------
    ```python
    >>> idl_float("1.6D4")
    1.6e4
    ```
    """
    idl_str = idl_num.lower()
    return float(idl_str.replace("d", "e"))


def smoothing_filter(wl_in, flux_in, vsini):
    """
    Apply smoothing filter to spectrum

    :param wl_in:
    :type wl_in:
    :param flux_in:
    :type flux_in:
    :return:
    :rtype:
    """
    if vsini == 0.0:
        warnings.warn("No smoothing applied when the")
        return wl_in, flux_in
    c = 299792.458
    f = 0.25  # arbitrary parameter
    w = vsini * f / c * np.mean(wl_in) / np.mean(np.diff(wl_in))

    kernel = Gaussian1DKernel(w)

    conv_flux = convolve(flux_in, kernel, boundary="extend")

    return conv_flux


def remove_continuum(wl_in, flux_in):
    """
    Remove continuum from spectrum

    :param wl_in:
    :type wl_in:
    :param flux_in:
    :type flux_in:
    :return:
    :rtype:
    """
    spectrum = Spectrum1D(
        flux=flux_in * u.erg / (u.s * u.cm ** 2),
        spectral_axis=wl_in * u.um,
    )

    g1_fit = fit_generic_continuum(spectrum, fitter=LinearLSQFitter())

    y_continuum_fitted = g1_fit(wl_in * u.um)

    spec_normalized = spectrum / y_continuum_fitted

    wl_in = spec_normalized.spectral_axis.value
    norm_spec = spec_normalized.flux.value

    return wl_in, norm_spec, y_continuum_fitted.value


def load_template(
        teff, logg, rpow, model_dir=None, template_dir=None, bounds=None, vsini=0.0
):
    if bounds:
        bstr = f"{bounds[0]}_{bounds[1]}"
    else:
        bstr = "None"

    if template_dir is not None:
        template_dir = Path(template_dir)
        pl_str = f"planet_teff_{teff}_logg_{logg}_vsini_{vsini}_bounds_{bstr}.csv"
        template_file = template_dir / pl_str
        if template_file.is_file():
            return np.loadtxt(template_file, delimiter=",", unpack=True)

    elif model_dir is None:
        raise ValueError("model_dir or template_dir needed")

    wl, fl = load_cifist(teff, logg, 0.0, 0.0, model_dir)
    wl, fl = get_wl_range(wl, fl, bounds=bounds)

    wl, _, fl = convolution(wl, fl, vsini, rpow)

    if template_dir is not None:
        data = np.array([wl, fl]).T
        np.savetxt(template_file, data, delimiter=",")


    return wl, fl

def get_file_name(name, info):
    # Get params from yaml
    teff = info["teff"]
    logg = info["logg"]
    vsini = info["vsini"]
    snr = info["snr"]

    # Define file name right now to check if need to run the simulation
    star_str = f"star_{name}_teff_{teff}_logg_{logg}_vsini_{vsini}_snr_{snr}"

    pl_str = ""
    companions = info["companions"] if "companions" in info else dict()
    for companion, cdict in companions.items():
        teff_pl = cdict["teff"]
        logg_pl = cdict["logg"]
        vsini_pl = cdict["vsini"]
        dv = cdict["dv"]
        cr = cdict["cr"]
        pl_str += f"_planet_{companion}_teff_{teff_pl}_logg_{logg_pl}_vsini_{vsini_pl}_cr_{cr}_dv_{dv}"

    fname = f"{star_str}{pl_str}.csv"

    return fname


def read_amescond(path):
    data = []
    with open(path) as data_file:
        for line in data_file:
            if '---' in line or line == '/n' or 'Isochrones' in line or 'Baraffe' in line or line.startswith('-') or 'Evolutionary' in line or 'case' in line or 'Filtres' in line:
                continue
            else:
                data.append(list(filter(None, line.rstrip().split(' '))))

    amescond = []
    for line in data:
        if '(Gyr)' in line:
            age = line[-1]

        elif 'M/Ms' in line:
            line.insert(0,'Age/Myr')
            header_cond = line[0:6]
        else:
            line.insert(0, age)
            amescond.append(line[0:6])

    header_cond = np.asarray(header_cond, dtype=str)
    amescond = np.asarray(amescond, dtype=np.float64)
    amescond[:, 0] *= 1e3  # (Myr)
    amescond[:, 1] *= (const.M_sun/const.M_jup).value  # (Mjup)
    amescond[:, 5] *= (1e9 * u.cm).to("Rjup").value  # (Rjup)
    header_cond[1]=header_cond[1].replace('s','j')
    header_cond[5]='R/Rj'

    return pd.DataFrame(amescond, columns=header_cond)


def read_bcah15(path):
    data = []
    with open(path) as data_file:
        for line in data_file:
            if '---' in line or line == '\n' or 'BAH15' in line or ':' in line or line.startswith('-'):
                continue
            else:
                data.append(list(filter(None, line.rstrip().split(' '))))
    del data[0:8] # remove headline

    bah = []
    for line in data:
        if '(Gyr)' in line:
            age = line[-1]
        
        elif 'M/Ms' in line:
            line.insert(1,'Age')
            header_bah = line[1:7]
        else:
            line.insert(0, age)
            bah.append(line[0:6])

    header_bah = np.asarray(header_bah, dtype=str)
    bah = np.asarray(bah, dtype=np.float64)
    bah[:, 0] *= 1e3  # (Myr)
    bah[:, 1] *= (const.M_sun/const.M_jup).value  # (Mjup)
    bah[:, 5] *= (const.R_sun/const.R_jup).value  # (Rjup)
    header_bah[1]=header_bah[1].replace('s','j')
    header_bah[5]=header_bah[5].replace('s','j')

    return pd.DataFrame(bah, columns=header_bah)

def find_closest_point(grid, mass, age, label=None):
    adiff = np.abs(grid["Age/Myr"] - age)
    grid_age = grid[adiff == adiff.min()]

    # TODO: Interpolate instead
    mdiff = np.abs(grid_age["M/Mj"] - mass)
    point = grid_age[mdiff == mdiff.min()]

    return point if label is None else point[label].item()

def save_to_txt(wls, spectrum, errors, wlunit="um", fname = "data.txt"):
    assert np.shape(wls) == np.shape(spectrum)
    assert np.shape(spectrum) ==  np.shape(errors)
    assert np.size(np.shape(wls)) == 1

    if wlunit=="um":
        wls = wls*1000
    elif wlunit=="angstrom" or wlunit=="a":
        wls = wls/10

    data = np.zeros((np.size(wls), 3))
    data[:,0] = wls
    data[:,1] = spectrum
    data[:,2] = errors

    np.savetxt(f"ispec_data/{fname}", data, delimiter=" ", header="waveobs\tflux\terr", comments='')
    

if __name__=="__main__":
    pass
##    wl_btsettl, fl_btsettl = load_btsettl(420, 4.5, 0.0, 0.0, \
##                                          Path("D:/Documents/Maitrise/bt-settl/2011/BT-Settl_M-0.0_a+0.0/"),\
##                                          wl_range=(0.5,2.5), norm=False)
##    plt.figure()
##    plt.plot(wl_btsettl, fl_btsettl)
##
##    wl_phoenix, fl_phoenix = load_phoenix(3600, 1.00, 2.0, 0.40, \
##                                          Path("D:/Documents/Maitrise/phoenix/spectres_test/PHOENIX-ACES-AGSS-COND-2011/"), \
##                                          phoenix_wav_path=Path("D:/Documents/Maitrise/phoenix/spectres_test/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"), \
##                                          wl_range=(0.5,2.5))
##    plt.figure()
##    plt.plot(wl_phoenix, fl_phoenix)
##    plt.show()
