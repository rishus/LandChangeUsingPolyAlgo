#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Original version created on Sept 23 05:10:47 2015
Fortran version is a ditto copy of this version. Has
been pretty robust till now -- well tried and tested.

This version created on Oct 6 19:18:41 2016.

@author: rishu
"""

#I'm starting this code from rs_map_v10.py
#-- This is the time from where on, bfast, ewmacd, and landtrendr will grow 
#   together in sync with each other. They will all have same inputs, I will
#   compare and contrast their methodologies, result metrics to develop my 
#   polyalgorithm. 

import numpy as np
import pywt
#from params import *
# if plotting:
#from matplotlib import pyplot as plt
#import datetime as dt
#import matplotlib.ticker as ticker
#import matplotlib.dates as mdates
#import os

"""
NA :  missing data
0  :  clear land
1  :  clear water
2  :  cloud shadow
3  :  snow
4  :  cloud
"""
def lsfit(t_loc, u, K, xbarlimit1):  #, plot):
    #t_loc is the training t
    """
    Least squares calculation for Step 1 of the pseudocode.
    
    Note that there are two steps of regression: First, regression is done over all (available)
    training data. Then, 'outliers' are filtered out and regression is done again on the 
    outlier free training data.
    
    Inputs:
        t_loc:       x-coordinates
        u:           y-coordinates (observations)
        K:           number of harmonics
        xbarlimit1:  parameter to determine 'outliers'.
                     (If the pointwise error is greater than this many standard deviations,
                     that point is considered an outlier.)
    
    Output:
        alpha_star:   numpy array of 2K+1 coefficents.
    
    """

    M = len(u)
    X = np.zeros((M, 2*K + 1))
    X[:,0] = 1
    for j in range(1, K+1):
        ccol = K + j
        X[:, j] = np.sin(list(map(lambda x: x * j,  t_loc[0:M])))   #np.sin(j * t_loc[0:M])
        X[:, ccol] = np.cos(list(map(lambda x: x * j,  t_loc[0:M])))   #np.cos(j * t_loc[0:M])

    if (np.abs(np.linalg.det(np.dot(np.transpose(X), X))) < 0.001):
        alpha_star = []
        return alpha_star 
    
    alpha = np.linalg.solve(np.dot(np.transpose(X), X), np.dot(np.transpose(X), u))
    Ealpha = u - np.dot(X, alpha)
    sigma = np.sqrt(float(M)/float(M-1)) * np.std(Ealpha)
    tau1 = xbarlimit1*sigma
    I = np.where(np.abs(Ealpha[0:M]) < tau1)[0] #[i for i in range(0, M) if np.abs(Ealpha[i]) < tau1]
    #I = [i for i in range(0, M) if np.abs(Ealpha[i]) < tau1]
    if (len(I) <= (2*K + 1)):
        alpha_star = []
        return alpha_star

    Xsub = X[I,:]
    alpha_star = np.linalg.solve(np.dot(np.transpose(Xsub), Xsub), np.dot(np.transpose(Xsub), np.asarray(u)[I]))

    return alpha_star

def getResiduals(alpha_star, t_loc, D, u, K, xbarlimit1, xbarlimit2, lowthreshold):
#t_loc is the full t (i.e., the present timepoints)
    """
    Residuals calculation for Step 1 of the pseudocode.
    
    Inputs:
        alpha_star:        refined coefficients determined with repeated regression
        t_loc:             numpy array of x-coordinates
        D:                 numpy array of observations
        u:                 training period observations
        xbarlimit1:        parameter to determine 'outliers' in the initial phase of determining 
                           refined coefficients.
        xbarlimit2:        Parameter for calculating the threshold to determine outliers
                           when refined coefficients are being used.
        
    
    Outputs:
        Ibar:              list
                           of indices of outlier-free time series
        sigma_Ihat:        scalar. 
                           standard deviation of outlier-free training data errors
        nullFlag:          Booloean.
                           True if further processing cannot be carried out. (E.g., if outlier-free-training data 
                           is of length less than 2K+1).
                           False otherwise.
        Estar_alphastar:   numpy array 
                           containing residuals (errors) for the entire time series
    
    """
    
    nullFlag = False
    S = len(D)
    M = len(u)
    Xbar = np.zeros((S, 2*K + 1))
    Xbar[:,0] = 1
    for j in range(1, K+1):
        ccol = K + j
        Xbar[:, j] = np.sin(list(map(lambda x: x * j,  t_loc)))   #np.sin(j * t_loc[0:M])
        Xbar[:, ccol] = np.cos(list(map(lambda x: x * j,  t_loc)))   #np.cos(j * t_loc[0:M])

    Estar_alphastar = D - np.dot(Xbar, alpha_star)
    if (max(np.abs(Estar_alphastar)) > 1000000 ):
        nullFlag = True
        Estar_alphastar = [-2222 for i in range(S)]
        Ibar = []
#        reconstruction = []
        sigma_Ihat = 0
        return Ibar, sigma_Ihat, nullFlag, Estar_alphastar  #, reconstruction
        
#    mu = np.mean(Estar_alphastar[0:M])
    sigma2 = np.sqrt(float(M)/float(M-1)) * np.std(Estar_alphastar[0:M])  # we're taking the sample standard deviation; note that a[0:m] will be a[0],...,a[M-1], therefore length M
    
    tau1 = xbarlimit1 * sigma2
    tau2 = xbarlimit2 * sigma2
    Ihat =  np.asarray([s for s in range(M) if ((np.abs(Estar_alphastar[s]) < tau1) and (D[s] > lowthreshold)) ])  #2nd condition redundant
    if (len(Ihat) < 2*K + 1):
        nullFlag = True
        Estar_alphastar = [-2222] * S
        Ibar = []
#        reconstruction = []
        sigma_Ihat = 0
        return Ibar, sigma_Ihat, nullFlag, Estar_alphastar  #, reconstruction
        
    Ibar2 = np.asarray([s for s in range(M,S) if ((np.abs(Estar_alphastar[s]) < tau2) and (D[s] > lowthreshold))])
#    Ibar2 = np.where(np.logical_and(np.abs(Estar_alphastar[M:S]) < tau2, D[M:S] > lowthreshold))[0]

    if (len(Ihat) > 0 and len(Ibar2 > 0)):
        Ibar =  np.append(Ihat, Ibar2)
    elif (len(Ihat)==0 and len(Ibar2)>0):
        Ibar = Ibar2
    elif (len(Ihat)>0 and len(Ibar2)==0):
        Ibar = Ihat
    else:
        Ibar = []

    Estar_alphastar_Ihat = Estar_alphastar[Ihat] #[Estar_alphastar[i] for i in Ihat]
    sigma_Ihat = np.sqrt(float(len(Ihat))/float(len(Ihat)-1)) * np.std(Estar_alphastar_Ihat)
    
#    reconstruction = np.dot(Xbar, alpha_star)
    
    return Ibar, sigma_Ihat, nullFlag, Estar_alphastar #, reconstruction

#@profile    
def get_control_limits(sigma, L, lamd, mu, len_Ibar):
    
    """
    Step 2 of the pseudocode.
    
    A control chart is built.
    Inputs:
        sigma, L, lamd, mu, |\bar{I}|:  Statistical information about the 'outlier-processed' timeseries.

    Outputs:
        tau:    control chart limits vector.
        
    """
    
    #tau = [0] * len_Ibar #for i in range(len_Ibar)]
    tau  = np.arange(len_Ibar)
    sl = sigma * L
    f = lamd/(2 - lamd)
    a = 1 - lamd
    tau = list(map(lambda x: mu + sl * np.sqrt(f * (1 - pow(a, 2*(x+1)))),  tau))
    return tau


def get_EWMA(Ibar, lamd, Estar_alphastar):
    
    """
    Step 3 of the pseudocode.
    
    EWMA of residuals resulting from harmonic regression is calculated.

    Inputs:
        Ibar:               indices of the outlier free timeseries
        lamd:               parameter used in the control chart calculations
        Estar_alphastar:    numpy array of the residuals of the time series

    Output:
        z:                  EWMA of residuals resulting from harmonic regression
    """
    
    z = [0 for i in range(len(Ibar))]
    z[0] = Estar_alphastar[Ibar[0]]
    a = 1.0 - lamd
        z[i] =  a * z[i-1]  + lamd * Estar_alphastar[Ibar[i]]
    
    return z
    for i in range(1, len(Ibar)):

    
def flag_history(z, tau, Ibar):
    """
    Step 4, flag history calculation of the pseudocode.
    
    Flag history is claculated here, based on the 'persistence' of any nonzero signals.
    
    Inputs:
        z:     EWMA of residuals resulting from harmonic regression
        tau:   Control chart vector
        Ibar:  Indices of outlier free timeseries.
    
    """
    
#    f = [int(np.sign(z[i]) * np.floor(np.abs(z[i]/tau[i]))) for i in range(len(z))]
    f = [(np.sign(z[i]) * np.floor(np.abs(z[i]/tau[i]))) for i in range(len(z))]
        
    return f
    
    
def persistence_counting(f, persistence):
    """
    Step 4, persistence calculation of the pseudocode
    Inputs:
        f:             the flag history
        persistence:   user defined parameter. (values between 7 and 10 have been used so far)
        
    Output:
        tmp4:          list of integers
                       each element indicates how long the current sign (+ or -) lasted.
    
    """
    
    f_sgn = np.sign(f)   # Disturbance direction
    s = len(f_sgn)
    shift_points = [-1] + [i for i in range(s-1) if f_sgn[i] != f_sgn[i+1]] +  [s-1]
    
    # Count consecutive dates in which directions are sustained
    sustenance = np.zeros((s))
    for i in range(0, len(shift_points)-1):
        sustenance[shift_points[i]+1:shift_points[i+1]+1] = shift_points[i+1] - shift_points[i]

    # If sustained dates are sustained for long enough, keep; otherwise set to previous sustained state
    tmp4 = [0 for i in range(s)]
    for i in range(persistence, s):
        if ((sustenance[i] < persistence) and (max(sustenance[0:i]) >= persistence)):
            b = sustenance[0:i][::-1]
            i1 = len(b) - np.argmax(b) - 1
            tmp4[i] = f[i1]
        else:
            tmp4[i] = f[i]
    
    return tmp4
    

def summarize(jump_vals_presSten, presInd, num_obs, summaryMethod, tyeardoy):
    
    """
    Summarizing.
    
    Missing data points are addressed here.
    Breakpoints are determined here.
    
    Inputs:
        summaryMethod:  string ('annual_mean' or 'on-the-fly')
                        For getting breakpoints, choose 'on-the-fly'
    
    Output:
        brkpt:          list of breakpoints indices
        ewma_summary:   jump values including the missing data points.
        brkpt_summary:  
        
    
    """
    missing_data_vals = -2222
    ewma_summary = [missing_data_vals for i in range(num_obs)]
    tt = 0
    for i in range(len(presInd)):
        ewma_summary[presInd[i]] = jump_vals_presSten[tt]
        tt += 1
    # missing and outlier timepoints still have -2222

    # if the very first obs was missing/outlier, set it to zero
    if ewma_summary[0] == missing_data_vals:
        ewma_summary[0] = 0

    # replace the -2222s with the last available ewma value
    if ewma_summary[0] != missing_data_vals:
        for i in range(1, num_obs):
            if ewma_summary[i] == -missing_data_vals:
                ewma_summary[i] = ewma_summary[i-1]

    if summaryMethod == 'on-the-fly':
        winsz = 50
        brkpt = [0]
        brkpt_summary = [0 for i in range(num_obs)]
        i = 1
        # find isolated breakpoints
        while i < num_obs:
            keep = False
#            keep2 = False
            if ewma_summary[i] != ewma_summary[i-1]:
                # we've hit a breakpoint
                keep = True
                fv1 = ewma_summary[i-winsz]
                for j in range(max(0, i-winsz+1), i):
                    if abs(ewma_summary[j] - fv1) >= 0.001:
                        keep = False
                        break
                
#                keep2 = True
#                lv1 = ewma_summary[min(i+winsz, num_obs)]
#                for j in range(i, min(i+winsz, num_obs)):
#                    if abs(ewma_summary[j] - lv1) >= 0.001:
#                        keep2 = False
#                        break
                
            # if this point is either a starting or an ending point, keep it.
            # no need to ev
            if keep == True: # or keep2 == True:
                brkpt.append(i-1)
                brkpt_summary[i-1] = ewma_summary[i-1]
                brkpt_summary[i] = ewma_summary[i]
                i += winsz
#            elif keep2 == True:
#                brkpt.append(i)
#                brkpt_summary[i] = ewma_summary[i]
#                brkpt_summary[i+1] = ewma_summary[i+1]
#                i += winsz
            else:
                i += 1        
#        while ((len(brkpt) < 2) and (i < num_obs) ):
#            if ewma_summary[i] != ewma_summary[i-1]:
#                    brkpt.append(i-1)
#            i += 1
        brkpt.append(num_obs-1)
                
    # IF ANNUAL MEAN IS TO BE USED:
    if summaryMethod=='annual_mean':
        beginIdx = 0
        endIdx = max([i for i in range(num_obs) if \
                                   tyeardoy[i, 0] == tyeardoy[beginIdx,0]])
        while endIdx != num_obs-1:
            mean = np.mean(ewma_summary[beginIdx : endIdx + 1])
            for i in range(beginIdx, endIdx+1):
                ewma_summary[i] = mean
            beginIdx = endIdx+1
            endIdx = max([i for i in range(num_obs) if \
                               tyeardoy[i, 0] == tyeardoy[beginIdx,0]])

    return brkpt, ewma_summary, brkpt_summary
    

def ewmacd(tyeardoy, vec_obs, presInd, \
           K, xbarlimit1, xbarlimit2,  \
           lowthreshold, trainingStart, trainingEnd, mu, L, lam,  \
           persistence, summaryMethod, ns, nc, full_fig_name):
    """
    Algorithm EWMACD.
    
    Inputs:
        tyeardoy:             A 2 column matrix --- 1st column contains the years, the second
                               column contains the doys.
        vec_obs:              A 1 column array. It contains spectral values (including missing)
                               for a fixed pixel and fixed band, in same chronological order as tyeardoy
        presInd:              numpy array containing the indices where vec_obs has is valid.
        K:                    The number of harmonics. 
        lowthreshold:         Minimum allowable value for a spectral band. Must be >= 0.
        trainingStart:        First year (eg., 2009)
        trainingEnd:          Final year + 1 (eg., 2015 if the dataset ends with 2014)
        
    Outputs:
        this_band_resids:              Residuals of the EWMA fit. Not always needed but, for this individual 
                                       algorithm, may be helpful in describing the quality of fit.
        this_band_summary:             EWMACD flags. 
                                       Original output of the algorithm.
                                       Redundant as of now in the LULC with polyalgo sense.
        this_band_brkptsglobalIndex:   List of breakpoints, each breakpoint being represented by its global index.
        this_band_brkPtYrDoy:          List of breakpoints, each breakpoint being described as [year, doy].
        brkpt_summary:                 brkptsummary is needed in these 1D codes to make 1D plots. 
                                       But for 2D, it is redundant.

    """
    num_obs = len(vec_obs)

    # ************* develop the presInd vector ***********************
    
#    presInd = np.where(vec_obs > lowthreshold)[0]
    
    tyeardoy_idxs = np.where(np.logical_and(trainingStart<= tyeardoy[:,0], tyeardoy[:,0]< trainingEnd))[0]

    common_idx = list(set(tyeardoy_idxs).intersection(presInd))
    training_t = tyeardoy[common_idx, 1] #[tyeardoy[i, 1] for i in presInd if \

#    # Corner case:
#    if (len(training_t) < 2 * K + 1):
#        this_band_fit = [-2222] * (nc+ns+1) # for i in range(nc+ns+1)]
#        this_band_resids = [-2222] * num_obs #for i in range(num_obs)]
#        this_band_summary = [-2222] * num_obs #for i in range(num_obs)]
#        this_band_brkptsglobalIndex = [0, num_obs-1]
#        this_band_brkPtYrDoy = [tyeardoy[0,:], tyeardoy[num_obs-1,:] ]
#        brkpt_summary = [-2222]*num_obs
#        return  this_band_resids, this_band_summary, this_band_brkptsglobalIndex, \
#                     this_band_brkPtYrDoy , brkpt_summary
    
    #*************** prepare data ***********************************        

    D = vec_obs[presInd] #[vec_obs[i] for i in presInd]        # vec_obs_pres
    t = map(lambda x: x * 2 * np.pi/365,  tyeardoy[presInd, 1])
    u = vec_obs[common_idx]    
    Sfinal = len(D)     # length of present data

    #*********** actual processing starts here *******************

    # compute the harmonic coeficients for this band
    this_band_fit = lsfit(training_t, u, K, xbarlimit1)
    # Corner case:
    if (len(this_band_fit)==0):        
        this_band_fit = [-2222] * (nc+ns+1)# for i in range(nc+ns+1)]
        this_band_resids = [-2222]* num_obs# for i in range(num_obs)]
        this_band_summary = [-2222]*num_obs  # for i in range(num_obs)]
        this_band_brkptsglobalIndex = [0, num_obs-1]
        this_band_brkPtYrDoy = [tyeardoy[0,:], tyeardoy[num_obs-1, :] ]
        brkpt_summary = [-2222]*num_obs
        return this_band_resids, this_band_summary, this_band_brkptsglobalIndex, \
                              this_band_brkPtYrDoy, brkpt_summary
    
    # compute the three residuals for this band
    Ibar, sigma_Ihat, nullFlag, Estar_alphastar  =  \
                              getResiduals(this_band_fit, t, D, u, K, \
                                         xbarlimit1, xbarlimit2, lowthreshold)    #, recontruction = \

    if (nullFlag == True):
        this_band_fit = [-2222] * (nc+ns+1)# for i in range(nc+ns+1)]
        this_band_resids = [-2222]* num_obs# for i in range(num_obs)]                         
        this_band_summary = [-2222]* num_obs # for i in range(num_obs)]
        this_band_brkptsglobalIndex = [0, num_obs]
        this_band_brkPtYrDoy = [tyeardoy[0,:], tyeardoy[num_obs-1, :] ]
        brkpt_summary = [-2222]*num_obs
        return this_band_resids, this_band_summary, this_band_brkptsglobalIndex, \
                              this_band_brkPtYrDoy, brkpt_summary

    # get control limits
    tau = get_control_limits(sigma_Ihat, L, lam, mu, len(Ibar))

    # get EWMA
    z = get_EWMA(Ibar, lam, Estar_alphastar)
        
    # get flag history
    f = flag_history(z, tau, Sfinal, Ibar)
        
    # detect jumps : here, only in the data that is present and not-labeled-as-outlier
    persistenceVec = persistence_counting(f, persistence)
        
    # by orgsten I mean the stencil consisting of all the 'present data';
    # notice that "present data: = "good data" \union "outlier data"
    # This stencil does not include the time stamps for which the data was missing!
    # Valuewise: outlier locations get value -2222, good locations get value in jump_vals
    jump_vals_presSten = -2222 * np.ones(Sfinal,  dtype=np.int)
    jump_vals_presSten[Ibar] = persistenceVec

    # summary for this band for this pixel: NOW we include the 'missing' data points as well
    this_band_brkptsglobalIndex, this_band_summary, brkpt_summary = \
                              summarize(jump_vals_presSten, presInd, num_obs, \
                                    summaryMethod, tyeardoy)

    this_band_brkPtYrDoy = [tyeardoy[i,:] for i in this_band_brkptsglobalIndex]

#    this_band_brkPtYrDoy_summary = [0000.000] * num_obs
#    for i in this_band_brkptsglobalIndex:
#        this_band_brkPtYrDoy_summary[i] = tyeardoy[i,0]+ float(tyeardoy[i, 1])/1000.0


    # summarize residuals
    this_band_resids = [-2222] *  num_obs #for i in range(num_obs)]

#        for i in range(0, len(this_band_summary)):
#            this_pixel_summary.append(this_band_summary[i])

    return  this_band_resids, this_band_summary, \
            this_band_brkptsglobalIndex, this_band_brkPtYrDoy, brkpt_summary
#brkptsummary is needed in these 1D codes to make 1D plots. But for 2D, it is redundant.
