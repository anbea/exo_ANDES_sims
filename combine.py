import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def combine_spectrums(wl_star, star_spectrum, wl_companion, companion_spectrum, cr, chem_norm = None, \
                      return_planet_signal=False, u_adjust=False):
    assert not any(wl_star - wl_companion); wl_obs = wl_star

    if u_adjust:
        fl_obs = star_spectrum * wl_obs ##Pour ajuster les unités flux_density => flux
    else:
        fl_obs = star_spectrum

    if chem_norm:
        tot=chem_norm
    else:
        tot = np.mean(companion_spectrum[companion_spectrum>np.quantile(companion_spectrum, [0.05])])
    
    fl_scaled = companion_spectrum / tot * cr

    if u_adjust:
        fl_scaled = fl_scaled * wl_obs

    fl_obs = fl_obs * (1 + fl_scaled)


    fl_pl = fl_obs*fl_scaled

    if return_planet_signal:
        chem_norm_constant = tot # Normalisation qui va devoir être fed dans le MCMC
        return wl_obs, fl_obs*fl_scaled, chem_norm_constant
    else:
        return wl_obs, fl_obs
    

if __name__=="__main__":
    pass
