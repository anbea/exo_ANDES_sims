import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def smoothen_ratios(ratios, n = 500):
    interval_1 = ratios[:171690] #I, Y & J
    interval_2 = ratios[171690:228435] #H
    interval_3 = ratios[228435:] #K

    r_1 = np.zeros_like(interval_1)
    for i in range(n,len(interval_1)-n):
        r_1[i] = np.sum(interval_1[(i-n):(i+n)])/(2*n)

    r_1[0:n] = (r_1[n+1]-r_1[n])*(np.arange(0,n)-n) + r_1[n]
    r_1[-n:] = (r_1[-(n+1)]-r_1[-(n+2)])*(np.arange(1,n+1)) + r_1[-(n+2)]
        
    r_2 = np.zeros_like(interval_2)
    for i in range(n,len(interval_2)-n):
        r_2[i] = np.sum(interval_2[(i-n):(i+n)])/(2*n)

    r_2[0:n] = (r_2[n+1]-r_2[n])*(np.arange(0,n)-n) + r_2[n]
    r_2[-n:] = (r_2[-(n+1)]-r_2[-(n+2)])*(np.arange(1,n+1)) + r_2[-(n+2)]
        
    r_3 = np.zeros_like(interval_3)
    for i in range(n,len(interval_3)-n):
        r_3[i] = np.sum(interval_3[(i-n):(i+n)])/(2*n)

    r_3[0:n] = (r_3[n+1]-r_3[n])*(np.arange(0,n)-n) + r_3[n]
    r_3[-n:] = (r_3[-(n+1)]-r_3[-(n+2)])*(np.arange(1,n+1)) + r_3[-(n+2)]

    return np.concatenate((r_1, r_2, r_3))

def polynomial_fit_ratios(wls, ratios, order=4):
    wls_1a = wls[:54697]; interval_1a = ratios[:54697] #I
    wls_1b = wls[54697:118850]; interval_1b = ratios[54697:118850] #Y
    wls_1c = wls[118850:171690]; interval_1c = ratios[118850:171690] #J
    wls_2 = wls[171690:228435]; interval_2 = ratios[171690:228435] #H
    wls_3 = wls[228435:]; interval_3 = ratios[228435:] #K


    popt1a = np.polyfit(wls_1a, interval_1a, deg = order)
    poly_1a = np.poly1d(popt1a)
    r_1a = poly_1a(wls_1a)

    popt1b = np.polyfit(wls_1b, interval_1b, deg = order)
    poly_1b = np.poly1d(popt1b)
    r_1b = poly_1b(wls_1b)

    popt1c = np.polyfit(wls_1c, interval_1c, deg = order)
    poly_1c = np.poly1d(popt1c)
    r_1c = poly_1c(wls_1c)

    popt2 = np.polyfit(wls_2, interval_2, deg = order)
    poly_2 = np.poly1d(popt2)
    r_2 = poly_2(wls_2)

    popt3 = np.polyfit(wls_3, interval_3, deg = order)
    poly_3 = np.poly1d(popt3)
    r_3 = poly_3(wls_3)

    return np.concatenate((r_1a, r_1b, r_1c, r_2, r_3))   

def gen_IFU_ratios(ifu_data, coords_1, coords_2, smooth=True, hexspax=False, plot=False):
    ##Importer les longueurs d'onde
    IFU_wls = np.load("ifudata\wls_PAOLA_psfs.npy")

    dims = np.shape(ifu_data)
    #print(dims)
    if not hexspax:
        if coords_1 == 'peak':
            x1 = round(dims[1]-1/2)
            y1 = round(dims[2]-1/2)
        elif coords_1 != 'tot':
            x1 = coords_1[0]
            y1 = coords_1[1]
        if coords_2 == 'peak':
            x2 = round(dims[1]-1/2)
            y2 = round(dims[2]-1/2)
        elif coords_2 != 'tot':
            x2 = coords_2[0]
            y2 = coords_2[1]

        if coords_1 == 'tot':
            ratios = np.sum(ifu_data, axis=(1,2))/ifu_data[:,x2,y2]
        elif coords_2 == 'tot':
            ratios = ifu_data[:,x1,y1]/np.sum(ifu_data, axis=(1,2))
        else:
            ratios = ifu_data[:,x1,y1]/ifu_data[:,x2,y2]

    else: ##if hexspax
        if coords_2 == 'peak':
            s2 = 0
        else:
            s2 = coords_2
        if coords_1 == 'peak':
            s1 = 0
        else:
            s1 = coords_1
            
        if coords_1 == 'tot':
            ratios = ifu_data[:,s2]
            
        elif coords_2 == 'tot':
            ratios = ifu_data[:,s1]

        else:
            ratios = ifu_data[:,s1]/ifu_data[:,s2]
            
    if smooth:
        ratios = polynomial_fit_ratios(IFU_wls, ratios, order=6)

    if plot:
        plt.figure()
        plt.plot(ratios)

    return ratios


if __name__=='__main__':
    pass
