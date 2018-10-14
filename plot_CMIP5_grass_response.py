#!/usr/bin/env python

"""
Calculate grassland GPP/NPP response to CO2 for each CMIP5 model. Here we are
assuming that > 70% grass frac is a 100% grassland pixel.

I'm also plotting the observed line

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (12.10.2018)"
__email__ = "mdekauwe@gmail.com"


import os
import sys
import glob
import numpy as np
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as optimize
from lmfit import minimize, Parameters, report_fit, Model, conf_interval,
     printfuncs, conf_interval2d

# Stop in 700's in whole years, these data are 12 months per yr
#rate = 1.0
#start_co2 = 280.
#for t in range(1, 140 + 1):
#         co2_conc  = start_co2 * ((1 + rate / 100)**t)
#         print(t-1, co2_conc)
# 24 359.08095860482547
# 96 735.0743833481944

def func(x, a, b):
    return a * np.log(x) - b

def main():

    grass_path = "data/1pctCO2/grassFrac"
    gpp_path = "data/1pctCO2/gpp"
    npp_path = "data/1pctCO2/npp"
    models = ["BNU-ESM","GFDL-ESM2M","IPSL-CM5A-LR","IPSL-CM5B-LR",\
              "MPI-ESM-LR","MPI-ESM-P","GFDL-ESM2G","HadGEM2-ES",\
              "IPSL-CM5A-MR","MIROC-ESM","MPI-ESM-MR"]

    models = ["HadGEM2-ES",\
              "IPSL-CM5A-MR","MIROC-ESM","MPI-ESM-MR"]


    rate = 1.0
    start_co2 = 359.08095860482547
    en = 96
    st = 24
    ny = en - st
    co2_conc = []
    for t in range(1, ny + 1):
        co2_conc.append(start_co2 * ((1 + rate / 100)**t))

    fig = plt.figure(figsize=[9,6])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    #plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color'] = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor'] = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    ax = fig.add_subplot(111)

    for model in models:

        fname = "%s/grassFrac_%s_1pctCO2_r1i1p1_1_140_1pctCO2_regrid_setgrid.nc"\
                    % (model, model)
        fname = os.path.join(grass_path, fname)
        ds = xr.open_dataset(fname)

        en = 96
        st = 24
        nmonths = 12
        real_st = st * nmonths
        real_en = real_st + ((en - st) * nmonths)

        # 359 ppm - 735 ppm
        grass_frac = ds.grassFrac[real_st:real_en,:,:]

        fname = "%s/npp_%s_1pctCO2_r1i1p1_gC_m2_month_1_140_1pctCO2_regrid_setgrid.nc"\
                    % (model, model)
        fname = os.path.join(npp_path, fname)
        ds = xr.open_dataset(fname)
        # 359 ppm - 735 ppm
        npp = ds.npp[real_st:real_en,:,:]

        mainly_grass = np.where(grass_frac > 0.7, npp, np.nan)

        mainly_grass = np.nanmean(mainly_grass, axis=(1,2)) # avg over lat,lon
        #mainly_grass = np.nanmean(mainly_grass, axis=(1)) # avg over lat
        #mainly_grass = np.nanmean(mainly_grass, axis=(1)) # avg over lon

        mainly_grass = mainly_grass.reshape((en-st),12)
        mainly_grass = np.nansum(mainly_grass, axis=1) # annual sum

        ini_val = mainly_grass[0]
        response = ((mainly_grass / ini_val)-1.) * 100.

        ax.plot(co2_conc, response, label=model)


    df_obs = pd.read_csv("/Users/mdekauwe/research/FACE/analysis/GRASS/observations/GRASS_FACE_OBS.csv", skiprows=1)
    df_obs["ANPP_RR"] = np.log(1.0 + df_obs.ANPP / 100.)
    mean_response = (np.exp(df_obs.groupby("SITE").ANPP_RR.mean())-1.0)*100.0
    sigma_response = (np.exp(df_obs.groupby("SITE").ANPP_RR.std())-1.0)*100.0

    CO2_INC = df_obs.CO2_INC.unique()

    popt, pcov = optimize.curve_fit(func, CO2_INC, mean_response.values,
                                    sigma=sigma_response.values,
                                    #sigma=np.ones(len(sigma_response.values)),
                                    absolute_sigma=True)

    xrange = np.arange(np.min(CO2_INC), 360)
    y_fit2 = func(xrange, popt[0], popt[1])

    params = Parameters()
    params.add('a', value=np.random.rand())
    params.add('b', value=np.random.rand())

    mod = Model(func)
    #result = mod.fit(mean_response.values, weights=1.0/(sigma_response.values**2),
    #                 x=CO2_INC, a=np.random.rand(), b=np.random.rand())


    result = mod.fit(mean_response.values, weights=1.0/sigma_response.values,
                     x=CO2_INC, a=np.random.rand(), b=np.random.rand())


    a = result.params["a"].value
    b = result.params["b"].value


    xrange = np.arange(np.min(CO2_INC), 360)
    y_fit = func(xrange, a, b)

    xrange +=350

    ax.plot(xrange, y_fit, lw=3, ls="-", color="black", label="Observations")
    #ax1.plot(xrange, y_fit2, ls="-.", color="seagreen")

    ax.legend(numpoints=1, loc="best")
    ax.set_xlabel("CO$_2$ (\u03BCmol mol$^{-1}$)")
    ax.set_ylabel("Normalised NPP response to CO$_2$ (%)")


    odir = "/Users/mdekauwe/Desktop"
    fig.savefig(os.path.join(odir, "CMIP5_GPP.pdf"), bbox_inches='tight',
                pad_inches=0.1)



if __name__ == "__main__":

    main()
