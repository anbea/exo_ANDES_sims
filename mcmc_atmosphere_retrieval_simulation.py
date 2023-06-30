import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import time
import emcee
import corner

import spaxelify_spectrum as spax
from combine import combine_spectrums


def log_prior(theta):
    cr = theta[0]
    dv = theta[1]
    if 1E-10 <= cr <= 1 and 1E-1 <= dv <=2E2:
        return -np.log(1/cr)
    else:
        return -np.inf

def log_likelihood(theta, star_data, pl_data, wl, data_fl, fl_err, \
                   time_multiplier, bandstr, fl_ref, fl_obs, spax_size, chem_norm_constant):

    cr = theta[0]
    dv = theta[1]

    wl_star = star_data[:,0]
    star_spectrum = star_data[:,1]
    pl_data = spax.process_pl_spectrum(pl_data[:,0], pl_data[:,1], wl_star, dv, spax_size=spax_size)
    
    wl_companion = pl_data[:,0]
    companion_spectrum = pl_data[:,1]
    
    wl_model, fl_model = combine_spectrums(wl_star, star_spectrum, wl_companion, companion_spectrum, cr, \
                                           chem_norm = chem_norm_constant, \
                                           return_planet_signal=False)
    #                                      return_planet_signal=True)
    #Pour éviter les erreurs de -2.7E-18
    fl_model = np.clip(fl_model, 0, None)
    ## Ce modèle (fl_model) est bruité par le spectrum stellaire utilisé. Son erreur est:
    fl_model_err = np.sqrt(fl_model)
    ## Maintenant que les spectres sont combinés, il faut les couper bande par bande et les rectifier
    
    data = {}
    for band in bandstr:
        data[band] = {}
        data[band]["wls"], data[band]["fl_model"] = spax.cut_band(wl_model, fl_model, band=band)
        wlcut, data[band]["fl_ref"] = spax.cut_band(wl, fl_ref, band=band)

        rapportM = data[band]["fl_model"]/data[band]["fl_ref"]

        scale = 0.03
##        scale = 0.01
        filt_dataM = spax.low_pass_convolution(rapportM, scale=scale, kern='sinc')
        if band == 'I':
            trimlen = 3500
        else:
            trimlen = 3000
        #trimlen = 10000
        
        data[band]["fl_model"] = data[band]["fl_model"][trimlen:-trimlen]
        data[band]["fl_ref"] = data[band]["fl_ref"][trimlen:-trimlen]
        rapportM = rapportM[trimlen:-trimlen]
        filt_dataM = filt_dataM[trimlen:-trimlen]
        filt_dataM_norm = filt_dataM/np.median(filt_dataM)*np.median(rapportM)

        ## Correction
        corr_flref = data[band]["fl_ref"]*filt_dataM_norm

        ## Normalisation du modèle
        wlcut, cut_fl_obs = spax.cut_band(wl, fl_obs, band=band) ## On normalise p/r à ça
        data[band]["fl_model"] = data[band]["fl_model"]/np.mean(data[band]["fl_model"])*np.mean(cut_fl_obs)
        corr_flref = corr_flref/np.mean(corr_flref)*np.mean(cut_fl_obs)

        ## Traitement du bruit sur le modèle
        data[band]["err_wls"], data[band]["fl_model_err"] = spax.cut_band(wl_model, fl_model_err, band=band)
        data[band]["fl_model_err"] = data[band]["fl_model_err"][trimlen:-trimlen]
##        print(f"fl_model_err, juste apres la combinaison: {np.log10(np.mean(data[band]['fl_model_err']))}")
        data[band]["fl_model_err"] = data[band]["fl_model_err"]*filt_dataM_norm
##        print(f"fl_model_err, apres mult avec filt: {np.log10(np.mean(data[band]['fl_model_err']))}")
        data[band]["fl_model_err"] = data[band]["fl_model_err"]/np.mean(data[band]["fl_ref"])*np.mean(cut_fl_obs)

        ## DIFFÉRENCE POUR LE MODÈLE
        data[band]["diffM"] = data[band]["fl_model"] - corr_flref


    fl_model = np.concatenate(tuple([data[band]["diffM"] for band in bandstr]))
    fl_model_err = np.concatenate(tuple([data[band]["fl_model_err"] for band in bandstr]))
    trimwls = np.concatenate(tuple([data[band]["wls"][trimlen:-trimlen] for band in bandstr]))


    fl_err = np.sqrt(fl_err**2 + fl_model_err**2)

    logl = -np.sum(np.log(fl_err**2*(2*np.pi))/2+(fl_model-data_fl)**2/(2*fl_err**2))
    return logl

def log_probability(theta, star_data, pl_data, wl, fl, fl_err, \
                    time_multiplier, bandstr, fl_ref, fl_obs, spax_size, chem_norm_constant):
    logprior = log_prior(theta)
    #print(np.random.randint(0,100))
    if not np.isfinite(logprior):
        return -np.inf
    else: 
        return log_likelihood(theta, star_data, pl_data, wl, fl, fl_err, \
                              time_multiplier, bandstr, fl_ref, fl_obs, spax_size, chem_norm_constant) + logprior

if __name__=='__main__':
    if 1:

        ##############################################
        import parameters ############################
        ##############################################

        cr = parameters.cr
        cr_th = cr
        dv = parameters.dv
        dv_th = dv 
        exo_spax = parameters.exo_spax
        spax_size = parameters.spax_size
        exoplanet = parameters.name
        bandstr = parameters.bandstr

        wl_companion = np.load(parameters.atm_wls_path)
        chem_data = np.load(parameters.spec_c_atm_path) ##PAS SPAXELIFIÉ
        pl_data_dv0 = np.zeros((np.size(chem_data), 2))
        pl_data_dv0[:,0] = wl_companion.copy()
        pl_data_dv0[:,1] = chem_data.copy()

        star_file = parameters.star_file
        time_multiplier = parameters.total_t_exp/parameters.base_t_exp # À cause de comment time_multiplier est défini

        wl_c_file = parameters.atm_wls_path
        spec_c_file = parameters.full_atm_path
        
        wl, fl_obs, fl_ref, pl_spec, fl_M, blank, chem_norm_constant = spax.generate_data(star_file, wl_c_file, spec_c_file, \
                                                          cr, time_multiplier, dv, spax_size, exo_spax, noise=True)


        stel_sub_data = {}
        for band in bandstr:

            ## Découpage des données (correction se fait bande par bande)
            wlcut, spec1cut = spax.cut_band(wl, fl_obs, band=band)
            err1 = np.sqrt(spec1cut)
            wlcut, spec2cut = spax.cut_band(wl, fl_ref, band=band)
            err2 = np.sqrt(spec2cut)

            rapport = spec1cut/spec2cut
            
            ## Filtre passe-bas
            stel_sub_data[band] = {}

            scale = 0.03
##            scale = 0.01
            filt_data = spax.low_pass_convolution(rapport, scale=scale, kern='sinc')
##            print(np.size(filt_data))
            if band == 'I':
                trimlen = 3500
            else:
                trimlen = 3000

            ## Enlever les effets de bords
            stel_sub_data[band]["wlsuntrimmed"] = wlcut.copy()
            stel_sub_data[band]["wls"] = wlcut[trimlen:-trimlen]
            stel_sub_data[band]["rapport"] = rapport[trimlen:-trimlen]
            stel_sub_data[band]["filt"] = filt_data[trimlen:-trimlen]
            stel_sub_data[band]["fspaxuntrimmed"] = spec1cut.copy()
            stel_sub_data[band]["fspax"] = spec1cut[trimlen:-trimlen]        
            stel_sub_data[band]["frefuntrimmed"] = spec2cut.copy()
            stel_sub_data[band]["fref"] = spec2cut[trimlen:-trimlen]
            stel_sub_data[band]["spaxnoise"] = np.sqrt(stel_sub_data[band]["fspax"])
            stel_sub_data[band]["refnoise"] = np.sqrt(stel_sub_data[band]["fref"])

            ##Normalisation
            stel_sub_data[band]["filt"] = stel_sub_data[band]["filt"]/np.median(stel_sub_data[band]["filt"])*np.median(stel_sub_data[band]["rapport"])

            ##Correction
            corr_ref = stel_sub_data[band]["fref"]*stel_sub_data[band]["filt"]

            ##Erreur
            stel_sub_data[band]["err"] = np.sqrt(stel_sub_data[band]["fspax"] + \
                                                 stel_sub_data[band]["fref"]*(stel_sub_data[band]["filt"])**2)

            ########### DONNÉES ###########
            stel_sub_data[band]["diff"] = stel_sub_data[band]["fspax"] - corr_ref
            

        wls = np.concatenate(tuple([stel_sub_data[band]["wls"] for band in bandstr]))
        wls_untrimmed = np.concatenate(tuple([stel_sub_data[band]["wlsuntrimmed"] for band in bandstr]))
        fl_obs = np.concatenate(tuple([stel_sub_data[band]["fspax"] for band in bandstr]))
        fl_obs_untrimmed = np.concatenate(tuple([stel_sub_data[band]["fspaxuntrimmed"] for band in bandstr]))
        fl_ref = np.concatenate(tuple([stel_sub_data[band]["fref"] for band in bandstr]))
        fl_ref_untrimmed = np.concatenate(tuple([stel_sub_data[band]["frefuntrimmed"] for band in bandstr]))
        fl_err = np.concatenate(tuple([stel_sub_data[band]["err"] for band in bandstr]))
        diff = np.concatenate(tuple([stel_sub_data[band]["diff"] for band in bandstr]))


        print("SIMULATION EN COURS")
        print(f"Exoplanet: {exoplanet} ")
        print(f"cr = {cr_th}")
        print(f"dv = {dv_th}")
        print(f"exposure_time = {parameters.total_t_exp}")
        print(f"bandes = {bandstr}")

        ## Résolution avec MCMC
        theta_th = np.array([cr, dv])
        nbwalkers = 6
        nbparams = len(theta_th)

        pos_init = theta_th + np.random.randn(nbwalkers, nbparams)*theta_th/100


        ## La star_data que j'ai besoin pour le modèle, dans la vraie vie, va venir des spaxels ailleurs que l'exoplanète.
        ## Ce spectre est donc bruité. C'est ce spectre qui servira pour combiner le signal modèle de l'exoplanète.

        star_data2 = np.zeros((np.size(wl),2))
        star_data2[:,0] = wl
        star_data2[:,1] = blank
        ## pl_data_dv0 est processed dans log_probability, aka la spaxelification de ce spectre-là n'est pas un problème

        ## fl doit être le signal rectifié du spaxel avec l'exoplanète, moins le signal de correction, normalisé
        ## bande par bande        
        plt.show()

##        print(np.mean(fl_err))
        
        sampler = emcee.EnsembleSampler(nbwalkers, nbparams, log_probability, \
                                        args=(star_data2, pl_data_dv0, wls_untrimmed, diff, fl_err, \
                                              time_multiplier, bandstr, fl_ref_untrimmed, fl_obs_untrimmed, \
                                              spax_size, chem_norm_constant))
        mcmc = sampler.run_mcmc(pos_init, 5000, progress=True)

    #Plotting and stuff
    fig, axes = plt.subplots(nbparams, figsize=(12, 4), sharex=True)
    samples = sampler.get_chain()
    labels = ["cr", "dv"]

    np.save(parameters.output_path, samples)

    for i in range(nbparams):
        ax = axes
        ax[i].plot(samples[:, :, i], "k", alpha=0.3)
        ax[i].set_xlim(0, len(samples))
        ax[i].set_ylabel(labels[i])
        ax[i].yaxis.set_label_coords(-0.1, 0.5)
    axes[i].set_xlabel("step number")

    #print(tau := sampler.get_autocorr_time())
    flat_samples = sampler.get_chain(discard=200, flat=True)

    #Corner plot emcee
    fig = corner.corner(flat_samples, labels=labels)
    plt.show()
