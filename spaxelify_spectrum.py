import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
import time

from combine import combine_spectrums
import utils as ut

import read_ifudata as rifu


def generate_data(spectrum_file, wl_companion_file, chem_data_file, cr, time_multiplier, dv, spax_size, \
                  exo_spax, dv_model = None, noise=False):
    wl_filename = "ifudata/wls_PAOLA_psfs.npy"
    full_IFU_data = f"ifudata/ifu_PAOLA_psfs_{spax_size}_spaxels.npy"
    ifu_wls, ifu_data = import_IFU_ratios(wl_filename, full_IFU_data)
    if 'hex' in spax_size:
        hexspax = True

    ## Importation spectre
    wl_spectrum, unspaxed_spectrum = import_spectrum(spectrum_file)
    unspaxed_spectrum = unspaxed_spectrum*time_multiplier
    ## Au cas où on a des longueurs d'onde en bas de 0.750
    unspaxed_spectrum = unspaxed_spectrum[wl_spectrum >= 0.7506]
    wl_spectrum = wl_spectrum[wl_spectrum >= 0.7506]

    ## Données exoplanète
    wl_companion = np.load(wl_companion_file)
    chem_data = np.load(chem_data_file) ##PAS SPAXELIFIÉ
    pl_data_dv0 = np.zeros((np.size(chem_data), 2))
    pl_data_dv0[:,0] = wl_companion.copy()
    pl_data_dv0[:,1] = chem_data.copy()

    ####################################################
    ## Préparation des spectres - SPECTRE DES DONNÉES ##
    ####################################################
    wl_spectrum_star, star_spaxed_spectrum1 = spaxelify_by_spax(wl_spectrum, unspaxed_spectrum, \
                                                                ifu_wls, ifu_data, exo_spax, cut=False, hexspax=hexspax)

    if not dv_model:
        dv_model = dv
    
    pl_data = process_pl_spectrum(wl_companion, chem_data, wl_spectrum_star, dv, spax_size=spax_size)
    pl_data_M = process_pl_spectrum(wl_companion, chem_data, wl_spectrum_star, dv_model, spax_size=spax_size)
    
    wl_companion = pl_data[:,0]
    companion_spectrum = pl_data[:,1]
    wl_companion_M = pl_data_M[:,0]
    companion_spectrum_M = pl_data_M[:,1]

    wl_spectrum1_uncut, spaxed_spectrum1_uncut = combine_spectrums(wl_spectrum_star, star_spaxed_spectrum1, \
                                                       wl_companion, companion_spectrum, cr)
    wl_spectrum1_uncut, pl_spectrum_uncut, chem_norm_constant = combine_spectrums(wl_spectrum_star, star_spaxed_spectrum1, \
                                                       wl_companion, companion_spectrum, cr, \
                                                              return_planet_signal=True)

    ## Pour éviter les erreurs du genre -1.6E-21,on clip les spectres pour les forcer à être > 0
    spaxed_spectrum1_uncut = np.clip(spaxed_spectrum1_uncut, 0, None)
    pl_spectrum_uncut = np.clip(pl_spectrum_uncut, 0, None)

    ## Noise
    if noise:
        try:
            spaxed_spectrum1 = np.random.poisson(lam=spaxed_spectrum1_uncut)
        except ValueError:
            spaxed_spectrum1 = np.random.normal(loc=spaxed_spectrum1_uncut, scale=np.sqrt(spaxed_spectrum1_uncut))

    ########################################################
    ## Préparation des spectres - SPECTRE DE RÉF + MODÈLE ##
    ########################################################
    wl_spectrum2, star_spaxed_spectrum2 = data_outside_spaxel(wl_spectrum, unspaxed_spectrum, ifu_wls, ifu_data, exo_spax, hexspax=hexspax)
    spaxed_blank = star_spaxed_spectrum2.copy()

    wl_spectrum2_uncut, spaxed_spectrum2_uncut = combine_spectrums(wl_spectrum_star, star_spaxed_spectrum2, \
                                                       wl_companion, companion_spectrum, 0)
    
    wl_spectrumM_uncut, spaxed_spectrumM_uncut = combine_spectrums(wl_spectrum_star, star_spaxed_spectrum2, \
                                                       wl_companion_M, companion_spectrum_M, cr)

    ## Encore une fois on veut les forcer à etre positifs
    spaxed_spectrum2_uncut = np.clip(spaxed_spectrum2_uncut, 0, None)
    spaxed_spectrumM_uncut = np.clip(spaxed_spectrumM_uncut, 0, None)

    ## Noise
    if noise:
        try:
            spaxed_spectrum2 = np.random.poisson(lam=spaxed_spectrum2_uncut)
##            spaxed_spectrumM = np.random.poisson(lam=spaxed_spectrumM_uncut)
        except ValueError:
            spaxed_spectrum2 = np.random.normal(loc=spaxed_spectrum2_uncut, scale=np.sqrt(spaxed_spectrum2_uncut))
##            spaxed_spectrumM = np.random.normal(loc=spaxed_spectrumM_uncut, scale=np.sqrt(spaxed_spectrumM_uncut))

        noise_ = spaxed_spectrum2 - spaxed_spectrum2_uncut
        spaxed_spectrumM = spaxed_spectrumM_uncut+noise_ ##Pour que le modèle ait le meme bruit que le flux fl_ref
                                                         ##qu'on a déjà. fl_model est calculé à partir de fl_ref
                                                         ##dans la vraie vie, ils doivent avoir le même bruit

        ## On s'assure une dernière fois d'avoir juste des strictement positives (pour pas de DIV0 error)
        spaxed_spectrumM = np.clip(spaxed_spectrumM, 1, None)
        spaxed_spectrum2 = np.clip(spaxed_spectrum2, 1, None)
        spaxed_spectrum1 = np.clip(spaxed_spectrum1, 1, None)
        spaxed_blank = np.clip(spaxed_blank, 1, None)
        pl_spectrum_uncut = np.clip(pl_spectrum_uncut, 1, None)
        
    assert not any(wl_spectrum2_uncut-wl_spectrum1_uncut)
    
    if noise:
        return wl_spectrum1_uncut, spaxed_spectrum1, spaxed_spectrum2, \
               pl_spectrum_uncut, spaxed_spectrumM, spaxed_blank, chem_norm_constant
    else:
        return wl_spectrum1_uncut, spaxed_spectrum1_uncut, spaxed_spectrum2_uncut, \
               pl_spectrum_uncut, spaxed_spectrumM_uncut, spaxed_blank, chem_norm_constant

def process_pl_spectrum(pl_wl, pl_fl, star_wl, dv, spax_size='hex0.01'):
    ## Tassage du spectre de la planète de dv et spaxelification de ce spectre
    if np.min(pl_wl) > 100: #aka si pl_wl est donné en nm
        pl_wl = pl_wl/1000
        
    wl_pl, fl_pl = ut.shift_spectrum(pl_wl, pl_fl, dv=dv)
    wl_pl, fl_pl = ut.get_wl_range(wl_pl, fl_pl, bounds=(np.min(star_wl)-0.05, np.max(star_wl)+0.05))
    
    fi = interp1d(wl_pl, fl_pl)
    fl_pl = fi(star_wl)

    wl_companion = star_wl; companion_spectrum = fl_pl

    ## Spaxelification
    ifu_wl_filename = "ifudata/wls_PAOLA_psfs.npy"
    ratios_filename = f"ifudata/ifu_ratio_peak_tot_{spax_size}.npy"
    ifu_wls, ifu_data = import_IFU_ratios(ifu_wl_filename, ratios_filename)

    wl_companion, companion_spectrum = spaxelify_spectrum(wl_companion, companion_spectrum, ifu_wls, ifu_data, \
                                                              cut=False)
    pl_data = np.zeros((np.size(wl_companion), 2))
    pl_data[:,0] = wl_companion
    pl_data[:,1] = companion_spectrum

    return pl_data


def spaxelify_spectrum(wl_spectrum, fl_spectrum, wl_IFU, ratios_IFU, reverse=False, cut=True):
    ## Créer la fonction d'IFU
    ifu_fct = interp1d(wl_IFU, ratios_IFU)

    ## Multiplication des affaires
    if not reverse:
        fl = fl_spectrum*ifu_fct(wl_spectrum)
    else:
        fl = fl_spectrum/ifu_fct(wl_spectrum)
    

    wl=wl_spectrum
    if cut:
        ## Mettre à 0 les entre-bandes
        fl[(wl>1.78394)*(wl<1.995)+(wl>1.32721)*(wl<1.47650)] = 0

    data = np.zeros((np.shape(wl_spectrum)[0], 2))
    data[:,0] = wl_spectrum
    data[:,1] = fl

    return wl_spectrum, fl


def cut_band(wl, fl, band=None):
    if np.min(wl) > 100:
        nm = True
        wls = wl/1000
    else:
        nm = False
        wls = wl
    if band == "I":
        r_spectrum = fl[np.logical_and((wls >= 0.750),(wls <= 0.950))]
        r_wls = wls[np.logical_and((wls >= 0.750),(wls <= 0.950))]
    elif band == "Y":
        r_spectrum = fl[np.logical_and((wls >= 0.950),(wls < 1.1135))]#1.1135
        r_wls = wls[np.logical_and((wls >= 0.950),(wls < 1.1135))]
    elif band == "J":
        r_spectrum = fl[np.logical_and((wls >= 1.1135),(wls <= 1.3272))]
        r_wls = wls[np.logical_and((wls >= 1.1135),(wls <= 1.3272))]
    elif band == "H":
        r_spectrum = fl[np.logical_and((wls >= 1.4765),(wls <= 1.78393))]
        r_wls = wls[np.logical_and((wls >= 1.4765),(wls <= 1.78393))]
    elif band == "K":
        r_spectrum = fl[np.logical_and((wls >= 1.995),(wls <= 2.38481))]
        r_wls = wls[np.logical_and((wls >= 1.995),(wls <= 2.38481))]
    elif band == "YJHK":
        indexing_array = ((wls>=0.9)&(wls<=1.3272))|((wls>=1.4765)&(wls<=1.78393))|((wls>=1.995)&(wls<=2.38481))
        r_spectrum = fl[indexing_array]
        r_wls = wls[indexing_array]
    elif band == "CY":
        r_spectrum = fl[np.logical_and((wls >= 1.0),(wls <= 1.01))]
        r_wls = wls[np.logical_and((wls >= 1.0),(wls <= 1.01))]
    elif band == "CJ":
        r_spectrum = fl[np.logical_and((wls >= 1.2000),(wls <= 1.21))]
        r_wls = wls[np.logical_and((wls >= 1.200),(wls <= 1.21))]
    elif band == "CH":
        r_spectrum = fl[np.logical_and((wls >= 1.6000),(wls <= 1.61))]
        r_wls = wls[np.logical_and((wls >= 1.600),(wls <= 1.61))]
    elif band == "CK":
        r_spectrum = fl[np.logical_and((wls >= 2.20),(wls <= 2.21))]
        r_wls = wls[np.logical_and((wls >= 2.20),(wls <= 2.21))]
    elif band == 'O2':
        r_spectrum = fl[np.logical_and((wls >= 1.25),(wls <= 1.28))] 
        r_wls = wls[np.logical_and((wls >= 1.25),(wls <= 1.28))]
    elif band == 'C':
        r_spectrum = fl[np.logical_and((wls >= 1.25),(wls <= 1.255))] 
        r_wls = wls[np.logical_and((wls >= 1.25),(wls <= 1.255))]
    elif band == 'HP1':
        r_spectrum = fl[np.logical_and((wls >= 0.7506),(wls <= 0.785))] 
        r_wls = wls[np.logical_and((wls >= 0.7506),(wls <= 0.785))]  
    elif band == 'HP2':
        r_spectrum = fl[np.logical_and((wls >= 0.756),(wls <= 0.773))] 
        r_wls = wls[np.logical_and((wls >= 0.756),(wls <= 0.773))]        
    else:
        r_spectrum = fl
        r_wls = wls
    if nm:
        r_wls = r_wls*1000

    return r_wls, r_spectrum    

def data_outside_spaxel(wl_spectrum, fl_spectrum, ifu_wls, ifu_data, spaxel, band=None, hexspax=False):
    wl_spax_spectrum, spax_data = spaxelify_by_spax(wl_spectrum, fl_spectrum, ifu_wls, ifu_data, spaxel, band=band, hexspax=hexspax)
    cut_wls, cut_fl = cut_band(wl_spectrum, fl_spectrum, band)


    assert not any(cut_wls-wl_spax_spectrum)
    
    remaining_spectrum = cut_fl - spax_data
    

    ## Il faut aussi enlever le spaxel du centre, qui sature
    wl_center_spectrum, center_spectrum = spaxelify_by_spax(wl_spectrum, fl_spectrum, ifu_wls, ifu_data, 0, band=band, hexspax=hexspax, cut=False)
    assert not any(wl_center_spectrum - wl_spax_spectrum)

    remaining_spectrum = remaining_spectrum - center_spectrum
    
    return wl_spax_spectrum, remaining_spectrum

def import_IFU_ratios(wl_filename, ratios_filename):
    ifu_wls = np.load(wl_filename)
    ifu_ratios = np.load(ratios_filename)

    return ifu_wls, ifu_ratios

def import_spectrum(filename):
    spectrum_data = np.load(filename)

    wl = spectrum_data[:,0]
    fl = spectrum_data[:,1]
    
    return wl, fl

def spaxelify_by_spax(wl_spectrum, fl_spectrum, ifu_wls, ifu_data, spaxel, reverse=False, band=None, \
                      cut=True, hexspax=False):

    if not hexspax and spaxel == 'peak':
        peak = round((np.shape(ifu_data)[1]-1)/2)
        spaxel=(peak, peak)
    
    spax_ifu_ratios = rifu.gen_IFU_ratios(ifu_data, spaxel, 'tot', hexspax=hexspax)
    

    wl_spectrum, spaxed_spectrum = spaxelify_spectrum(wl_spectrum, fl_spectrum, ifu_wls, spax_ifu_ratios, \
                                                      reverse=reverse, cut=cut)

    r_wl_spectrum, r_spectrum = cut_band(wl_spectrum, spaxed_spectrum, band)

    return r_wl_spectrum, r_spectrum


def low_pass_convolution(fl_spectrum, scale=0.01, kern='sinc'):
    size = np.size(fl_spectrum)

    bell_size = round(size*scale*3)
    if kern == 'rect':
        bell = np.ones(bell_size)
    elif kern == 'sinc':
        bell = np.sinc(np.linspace(-3*np.pi, 3*np.pi, bell_size))
    elif kern == 'norm':
        bell = norm.pdf(np.linspace(-bell_size, bell_size, 2*bell_size-1), loc=0, scale=(scale*size))
    norm = np.sum(bell)
    
    baseline = np.convolve(fl_spectrum, bell, mode='same')/norm

    return baseline


if __name__=='__main__':
    pass
    

