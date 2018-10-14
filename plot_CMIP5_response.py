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
from lmfit import minimize, Parameters, report_fit, Model, conf_interval

# Stop in 700's in whole years, these data are 12 months per yr
#rate = 1.0
#start_co2 = 280.
#for t in range(1, 140 + 1):
#         co2_conc  = start_co2 * ((1 + rate / 100)**t)
#         print(t-1, co2_conc)
# 24 359.08095860482547
# 96 735.0743833481944


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

        en = 96
        st = 24
        nmonths = 12
        real_st = st * nmonths
        real_en = real_st + ((en - st) * nmonths)

        fname = "%s/npp_%s_1pctCO2_r1i1p1_gC_m2_month_1_140_1pctCO2_regrid_setgrid.nc"\
                    % (model, model)
        fname = os.path.join(npp_path, fname)
        ds = xr.open_dataset(fname)
        # 359 ppm - 735 ppm
        npp = ds.npp[real_st:real_en,:,:]

        npp = np.nanmean(npp, axis=(1,2)) # avg over lat,lon

        npp = npp.reshape((en-st),12)
        npp = np.nansum(npp, axis=1) # annual sum

        ini_val = npp[0]
        response = ((npp / ini_val)-1.) * 100.

        ax.plot(co2_conc, response, label=model)


    ax.legend(numpoints=1, loc="best")
    ax.set_xlabel("CO$_2$ (\u03BCmol mol$^{-1}$)")
    ax.set_ylabel("Normalised NPP response to CO$_2$ (%)")


    odir = "/Users/mdekauwe/Desktop"
    fig.savefig(os.path.join(odir, "CMIP5_NPP.pdf"), bbox_inches='tight',
                pad_inches=0.1)



if __name__ == "__main__":

    main()
