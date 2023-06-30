## This file is used to define the parameters to be used in the MCMC simulations

###################################################################
## Exoplanet parameters
###################################################################

cr = 6E-5 ## LOCAL contrast;
          ## meaning the exoplanet's contrast within a given spaxel (the exo_spax spaxel).
          ## cr_local = cr_tot/cr_AO
          ## For Proxima b, cr = 1.35E-7/0.0024 ~= 6E-5

#Alternatively; manual computation of the local contrast. example with Proxima b

##import numpy as np
##A_b = 0.3 ## Albedo of the exoplanet
##radius = 1.1 * 6378100 ## Radius of the exoplanet in meters
##separation = 0.0496 * 1.496E11 ## Star-planet separation in meters
##phase_function = 0.5 ## Phas function (typically 0.5)
##
##cr_tot = A_b*(radius/separation)**2*phase_function
##
##cr_AO_data = np.load("ifudata/averaged_ifu_PSFS_hex0.01.npy")
##possible_proxb_spaxels = [7, 8, 9, 10, 17, 18, 19, 20, 27, 28, 29, 30, 37, 38, 39, 40, 47, 48, 49, 50, 57, 58, 59, 60]
##av_cr_AO = np.mean(cr_AO_data[possible_proxb_spaxels]) ##Average AO performances at the exoplanet's possible positions
##
##cr = cr_tot/av_cr_AO ## Returns 5.43E-5, not quite 6E-5 given higher for this system.
##                     ## The 6E-5 figure comes from a more thorough statistical
##                     ## analysis of the parameters of Proxima b with their uncertianties, but
                       ## the 5.43E-5 value is a fair approximation and remains a likely value.



dv = 30 ## Radial velocity, in km/s

exo_spax = 19 ## ID number of the spaxel in which the exoplanet is to be placed
              ## Refer to ID figure
spax_size = "hex0.01" ## IFU used for the observations:
                      ## Either hex0.005 for the 5-mas spaxel size IFU
                      ## or hex0.01 for the 10-mas spaxel size IFU




###################################################################
## Host star parameters
###################################################################

## Only required variable is the star spectrum path.
## Must be a .npy file of dimensions [N,2] where data[:,0] gives the wavelengths and data[:,1] gives the spectrum, in photons per unit time.
## The data is expected to already be given at a resolution of 100 000, with 3 pixels per resolution element
name = "proxima"
star_file = f"spectradata/{name}_20hours_IYJHK.npy"
base_t_exp = 20 ## Unit of time used to generate the data in star_file, in hours.


###################################################################
## Observation parameters
###################################################################

total_t_exp = 20 ## Total exposure time wanted in the search

bandstr = "YJH" ## Bands in which observations will be made; supported bands are I Y J H K


###################################################################
## Chemical data
###################################################################

## The data is expected to already be given at a resolution of 100 000, with 3 pixels per resolution element.

# Path to the wavelengths of the atmosphere spectrum. 
atm_wls_path = "spectradata/NASA_wls_IYJHK.npy" 

# Path to the absorbance of the full atmosphere. Must be >= 0 and <= 1.
full_atm_path = "spectradata/NASA_spectrum_IYJHK.npy"

# Path to the absorbance of the chemical we are currently searching for. Must be >= 0 and <= 1.
# May or may not be the same path as full_atm_path. Both absorbances must follow the same wavelengths.
spec_c_atm_path = "spectradata/NASA_H2O_IYJHK.npy"


###################################################################
## Output_path
###################################################################

## This saves the emcee samples in an npy file (first cr samples, then dv samples)
output_path = "results/test.npy"
