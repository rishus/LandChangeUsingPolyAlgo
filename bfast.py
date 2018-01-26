#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Original created on Sun Jul 31 13:10:33 2016
Original code is in from /bfast/python/bfast.py
That code was running complete n all. 

This version created on Thu Oct 6 13:10:33 2016.
@author: rishu
"""

import numpy as np
#import stl as st
import scipy
import scipy.linalg as la
import copy
from matplotlib import pyplot as plt
import sys
def critValuesTable(method, startRow, endRow):
    
    if (method == "brownian bridge increments"):
        
       BBI = [[ 0.7552,  0.8017,  0.8444,  0.8977], \
           [ 0.9809,  1.0483,  1.1119,  1.1888],   \
           [ 1.1211,  1.2059,  1.2845,  1.3767],   \
           [ 1.217 ,  1.3158,  1.4053,  1.5131],   \
           [ 1.2811,  1.392 ,  1.4917,  1.6118],   \
           [ 1.3258,  1.4448,  1.5548,  1.6863],   \
           [ 1.3514,  1.4789,  1.5946,  1.7339],   \
           [ 1.3628,  1.4956,  1.6152,  1.7572],   \
           [ 1.361 ,  1.4976,  1.621 ,  1.7676],   \
           [ 1.3751,  1.5115,  1.6341,  1.7808],   \
           [ 0.7997,  0.8431,  0.8838,  0.9351],   \
           [ 1.0448,  1.1067,  1.1654,  1.2388],   \
           [ 1.203 ,  1.2805,  1.3509,  1.4362],   \
           [ 1.3112,  1.4042,  1.4881,  1.5876],   \
           [ 1.387 ,  1.4865,  1.5779,  1.693 ],   \
           [ 1.4422,  1.5538,  1.653 ,  1.7724],   \
           [ 1.4707,  1.59  ,  1.6953,  1.8223],   \
           [ 1.4892,  1.6105,  1.7206,  1.8559],   \
           [ 1.4902,  1.6156,  1.7297,  1.8668],   \
           [ 1.5067,  1.6319,  1.7455,  1.8827],   \
           [ 0.825 ,  0.8668,  0.904 ,  0.9519],   \
           [ 1.0802,  1.1419,  1.1986,  1.27  ],   \
           [ 1.2491,  1.3259,  1.3951,  1.482 ],   \
           [ 1.3647,  1.4516,  1.5326,  1.6302],   \
           [ 1.4449,  1.5421,  1.6322,  1.747 ],   \
           [ 1.5045,  1.6089,  1.7008,  1.8143],   \
           [ 1.5353,  1.656 ,  1.751 ,  1.8756],   \
           [ 1.5588,  1.6751,  1.7809,  1.9105],   \
           [ 1.563 ,  1.6828,  1.7901,  1.919 ],   \
           [ 1.5785,  1.6981,  1.8071,  1.9395],   \
           [ 0.8414,  0.8828,  0.9205,  0.9681],   \
           [ 1.1066,  1.1663,  1.2217,  1.2918],   \
           [ 1.2792,  1.3533,  1.4212,  1.5013],   \
           [ 1.3973,  1.4506,  1.5593,  1.6536],   \
           [ 1.4852,  1.5791,  1.669 ,  1.7741],   \
           [ 1.5429,  1.6465,  1.742 ,  1.8573],   \
           [ 1.5852,  1.6927,  1.7941,  1.914 ],   \
           [ 1.6057,  1.7195,  1.8212,  1.945 ],   \
           [ 1.6089,  1.7245,  1.8269,  1.9592],   \
           [ 1.6275,  1.7435,  1.8495,  1.9787],   \
           [ 0.8541,  0.8948,  0.9321,  0.9799],   \
           [ 1.1247,  1.1846,  1.2395,  1.3088],   \
           [ 1.304 ,  1.3765,  1.444 ,  1.5252],   \
           [ 1.425 ,  1.5069,  1.5855,  1.6791],   \
           [ 1.5154,  1.6077,  1.6921,  1.7967],   \
           [ 1.5738,  1.677 ,  1.7687,  1.8837],   \
           [ 1.6182,  1.7217,  1.8176,  1.9377],   \
           [ 1.646 ,  1.754 ,  1.8553,  1.9788],   \
           [ 1.6462,  1.7574,  1.8615,  1.9897],   \
           [ 1.6644,  1.7777,  1.8816,  2.0085],   \
           [ 0.8653,  0.9048,  0.9414,  0.988 ],   \
           [ 1.1415,  1.1997,  1.253 ,  1.622 ],   \
           [ 1.3223,  1.3938,  1.4596,  1.5392],   \
           [ 1.4483,  1.5305,  1.61  ,  1.7014],   \
           [ 1.5392,  1.6317,  1.7139,  1.8154],   \
           [ 1.6025,  1.7018,  1.793 ,  1.9061],   \
           [ 1.6462,  1.7499,  1.8439,  1.9605],   \
           [ 1.6697,  1.7769,  1.8763,  1.9986],   \
           [ 1.6802,  1.7889,  1.8932,  2.0163],   \
           [ 1.6939,  1.8052,  1.9074,  2.0326]]  
   
       critValTable = BBI[startRow:endRow]

    return critValTable


def regress(t, u, model, K):
    
    M = len(t)
    if (model == 'linear') or (M < 2*K+1):
        ncols = 2
    elif (model == "harmon"):
        ncols = 2*K+1
    else:
        print 'model not supplied'
    
    X = np.zeros((M, ncols))
    X[:,0] = 1
        
    if (model == 'linear') or (M < 2*K+1):
        X[:, 1] = t
    elif (model == 'harmon'):
        for j in range(1, K+1):
            X[:, 2*j-1] = np.cos(map(lambda x: x * j,  t[0:M]))   #np.cos(j * t_loc[0:M])
            X[:, 2*j] = np.sin(map(lambda x: x * j,  t[0:M]))   #np.sin(j * t_loc[0:M])
#            X[:, 2*j-1] = np.asarray([np.cos(j * t[i]) for i in range(0,M)])
#            X[:, 2*j] = np.asarray([np.sin(j * t[i]) for i in range(0,M)])
    else:
        print "model not supported"
    
    if (np.abs(np.linalg.det(np.dot(np.transpose(X), X))) < 0.001):
        alpha_star = [0,0]
        fit = np.zeros((M,))
        return alpha_star, fit

    alpha = np.linalg.solve(np.dot(np.transpose(X), X), np.dot(np.transpose(X), u))
    fit = np.dot(X,alpha)
#    print 'fit_dims: ', fit.shape

    return alpha, fit
    

def OLSMosum(t, obs, model, h, K):
    
    Sfinal = len(t)
    if (model == "linear"):
        coeffs, fit = regress(t, obs, "linear", 0)
    elif (model == "harmon"):
        coeffs, fit = regress(t, obs, "harmon", 0)
    else:
        print "model not supported"
    
    residuals = fit - obs
    
    dofResiduals = Sfinal - 2   
    sigma = np.sqrt(float(Sfinal)/float(dofResiduals)) * np.std(residuals)  # we're taking the sample standard deviation
    nh = int(np.floor(Sfinal*h))
    cum_resids = np.cumsum(residuals)
    process = cum_resids[nh-1:] - np.concatenate(([0.0], cum_resids[0:Sfinal-nh]))
    process = process/(sigma*np.sqrt(Sfinal))

    return process


def calcPValue(x, method, k, h, fnal):
    #k is the number of columns in the 'process'
    #h ... i dont know. looks like some parameter related to grid spacing
            
    if (method == "brownian bridge increments"):
        if (k > 6):
            k=6
        start = (k-1)*10+1
        endr = k*10
        if (fnal == "max"):
            critValTable = critValuesTable("brownian bridge increments",start-1, endr)
            tableNcols = len(critValTable[1])  #redundant as of now becuz we've allocated fixen no. of cols = 4 in/for the table.
            tableH = [0.05*i for i in  range(1,10+1) ]
            tableP = [ 1.0, 0.1, 0.05, 0.02, 0.01 ]
            tableipl = [ 0 for i in range(0,tableNcols+1) ]
            start = [i for i in range(0, 9) if tableH[i]<=h and tableH[i+1] >= h ][0]
            endr = start+1
            for i in range(0, tableNcols):
                x1 = tableH[start]
                x2 = tableH[endr]
                y1 = critValTable[start][i]
                y2 = critValTable[endr][i]
                slope = (y2-y1)/(x2-x1)
                tableipl[i+1] = y1 + slope * (h - x1)
            if (x > tableipl[4]):
                pval = tableP[4]
            elif (x < tableipl[0]):
                pval = tableipl[0]
            else:
                start = [i for i in range(0,tableNcols) if tableipl[i]<= x and tableipl[i+1]>=x][0]
                endr = start + 1
                x1 = tableipl[start]
                x2 = tableipl[endr]
                y1 = tableP[start]
                y2 = tableP[endr]
                slope = (y2-y1)/(x2-x1)
                pval = y1 + slope * (x - x1)
    
    return pval
    
def recresids(t, u, begin_idx, model, deg):

    # remember, in Python, indices here will start from 0. 
    # begin_idx denotes the first point for which the residual will be calculated.
    # So remember use begin_idx as one less than what we were using in Fortran.
    # Also, begin_idx is only wrt the subset array that has been passed in.
    # It's not wrt the original global array. 
    Sfinal = len(t)
    if (model == 'linear'):
        ncols = 2
        X = np.zeros((Sfinal, ncols))
        X[:, 0] = 1
        X[:, 1] = t
    elif (model == 'harmon'):
        ncols = 2*deg + 1
        X = np.zeros((Sfinal, ncols))
        X[:, 0] = 1
        for j in range(1, deg+1):
            X[:, 2*j-1] = np.asarray([np.cos(j * t[i]) for i in range(0,Sfinal)])
            X[:, 2*j] = np.asarray([np.sin(j * t[i]) for i in range(0,Sfinal)])
    else:
        print "model not supported"    
        
    check = True
    recres = [ 0 for i in range(0, Sfinal) ]
    # begin_idx is the firstmost index where the residual will be calculated.
    # The lastmost residual will be calculate for the index Sfinal.
    # But python indices will run from 0 to Sfinal -1, Sfinal being the length of the current array.
    # So residuals will be calculated from indices begin_idx-1 to Sfinal-1
    # So, essentially, the residuals will be calculated from 
    # We have designated curr_idx as the point where the estimation window ends.
    # So curr_idx will run from begin_idx-1
    for curr_idx in range(begin_idx-1, Sfinal-1):
        if (check == True):    #we are setting it to be true forever, though.
#            q, r = np.linalg.qr(X)
#            p = np.dot(q.T, u)
#            vec_fitcoefs = np.dot(np.linalg.inv(r), p)
#            vec_fitobs = np.dot(X, vec_fitcoefs)
            A = np.dot(X[0:curr_idx+1,:].T, X[0:curr_idx+1,:])   #becuz the last index does not get included
            L = scipy.linalg.cho_factor(A, lower=False, overwrite_a=False,check_finite=True)
            vec_fitcoefs = scipy.linalg.cho_solve(L, np.dot(X[0:curr_idx+1,:].T, u[0:curr_idx+1]), overwrite_b=False, check_finite=True)
            rank = np.linalg.matrix_rank(L)   #uses svd
            
            if (rank > 2):
                print "rank > 2. Can't proceed. Cross check recresid."
                break
            if (rank == 0):
                print  "rank = 0 in recres"
                break
            
        else:
            i = 1

#       y = scipy.linalg.cho_solve(L, X[i+1,:].T, overwrite_b=False, check_finite=True)
        fr = 1 + X[curr_idx+1,:].dot(la.cho_solve(L, X[curr_idx+1,:].T, overwrite_b=False, check_finite=True))   #np.linalg.solve(r.T.dot(r) , X[i+1,:].T)
#        print  'fr =', fr
        recres[curr_idx + 1] = (u[curr_idx+1] - np.dot(X[curr_idx+1, :], vec_fitcoefs))/np.sqrt(fr)
    
    return recres


def getRSStri(t, u, model, h, K):
    
    # remember, unlike Fortran, indices here will start from 0. 
    # So remember use begin_idx as one less than what we were using in Fortran.
    # Basically, recresid will get filled from idx=ncols to idx=Sfinal for linear regression.
    # Fortran wud have filled it from idx = ncols+1 to Sfinal.

    # build RSS matrix
    if (model == 'linear'):
        ncols = 2
    elif (model == 'harmonic'):
        ncols = 2*K+1

    Sfinal = len(t)
    RSStri =  [[0 for i in range(0,Sfinal)] for j in range(0,Sfinal)]
    brkpt_spacing = int(np.floor(Sfinal * h))
    if brkpt_spacing <= ncols:
        print "minimum segment size must be greater than the number of regressors; resetting"
        brkpt_spacing = ncols + 2  #this number 2 is a random choice
        
    for idx in range(Sfinal- brkpt_spacing +1):
        if (model == 'linear'):
            tmp = recresids(t[idx:], u[idx:], ncols, 'linear', 1)
        elif (model == 'harmonic'):
            tmp = recresids(t[idx:], u[idx:], ncols, 'harmon', K) 
        else:
            print "model not supported"
        tmp2 = [i*i for i in tmp]
        RSStri[idx][idx:] = np.cumsum(tmp2)
        
    return RSStri


def buildDynPrTable(RSSTri_full, numBrks, Sfinal, h):
    """
    RSSTri_full[i,j]: cost of building series from i till j
    h: is a fraction.

    Result:
    A vector of breakpoints

    matCost[0,:] cost of 1 breakpoint and two signals [0:b0] and [b0:-1]
    matCost[1,:] cost of 2 breakpoint and three signals [0:b0] and [b0:b1] and [b1:-1]
    matCost[2,:] cost of 3 breakpoint and four signals [0:b0] and [b0:b1] and [b1:b2], [b2:-1]

    """
    #TODO: ; +1 to start index from 1
    matCost = np.zeros((numBrks, Sfinal))
    matPos = np.zeros((numBrks, Sfinal))

    matPos[:,:] = -1
    brkpt_spacing = int(np.floor(Sfinal*h))
#    print brkpt_spacing
#    print "brkpt_spacing = ", brkpt_spacing
    matCost[:,:] = sys.maxint
    matCost[0,brkpt_spacing:Sfinal-brkpt_spacing] = RSSTri_full[0][brkpt_spacing:Sfinal-brkpt_spacing]
    matPos[0,brkpt_spacing:Sfinal-brkpt_spacing] = [i for i in range(brkpt_spacing,Sfinal-brkpt_spacing)]
#    print matCost[0,brkpt_spacing:Sfinal-brkpt_spacing]
    for nbs in range(1,numBrks):
        beginIdx = nbs * brkpt_spacing  #obvious
        endIdx = Sfinal - brkpt_spacing #obvious
        for idx in range(beginIdx, endIdx):
            potIdxBegin = nbs * brkpt_spacing   #valid pos for nbs-1 breakpoints
            potIdxEnd = min(idx- brkpt_spacing, Sfinal-brkpt_spacing)
            #print "idx = ", idx, "potRange = ", potIdxBegin, ", ", potIdxEnd
            vecCost = [sys.maxint for i in range(0, Sfinal)]
            for j in range(potIdxBegin, potIdxEnd):
                vecCost[j] = matCost[nbs-1, j] + RSSTri_full[j][idx]
                #if nbs == 1:
                #    print "choices: bp = ", j, " ", matCost[nbs-1,j], " ", RSSTri_full[j][idx]
                assert j< idx

            matCost[nbs, idx]  = min(vecCost)
            matPos[nbs, idx] = int(np.argmin(vecCost))
#            print 'sys.maxint = ', sys.maxint
#            if idx == 76:
#                print  nbs, potIdxBegin, " ", potIdxEnd, " ", np.argmin(vecCost), vecCost
    
    for idx in range(numBrks * brkpt_spacing, Sfinal-brkpt_spacing):
        matCost[numBrks-1, idx] = matCost[numBrks-1, idx] + RSSTri_full[idx][-1]

#    print "numBrks = ", numBrks
    last_brkpt_pos = numBrks * brkpt_spacing + np.argmin(matCost[numBrks-1, numBrks * brkpt_spacing: Sfinal-brkpt_spacing])
#    print last_brkpt_pos
    curr_brkpt_pos = int(last_brkpt_pos)
    vecBrkPts = [-1 for i in range(0, numBrks)]
    vecBrkPts[-1] = int(last_brkpt_pos)
    i = numBrks - 1
    while(i > 0):
        curr_brkpt_pos = matPos[i, int(curr_brkpt_pos)] #start pos for last segment
#        print i, curr_brkpt_pos
        vecBrkPts[i-1] = int(curr_brkpt_pos)  #we are ignoring the 0-row,0-column
        i = i -1
#    print "Breakpoints = ", vecBrkPts

    return  vecBrkPts


def buildDynPrTable_stencil(RSSTri_full, numBrks, Sfinal, h):
    """
    RSSTri_full[i,j]: cost of building series from i till j
    h: is a fraction.

    Result:
    A vector of breakpoints
    """
    #TODO: ; +1 to start index from 1
    matCost = np.zeros((numBrks+1, Sfinal+1))
    matPos = np.zeros((numBrks+1, Sfinal+1))

    matPos[:,:] = -1
    brkpt_spacing = int(np.floor(Sfinal*h))
    
    matCost[:,:] = 99999999
#    print "Values read: [0]",brkpt_spacing-1,  ":", Sfinal-brkpt_spacing
    matCost[1,brkpt_spacing:Sfinal-brkpt_spacing+1] = RSSTri_full[0][brkpt_spacing-1:Sfinal-brkpt_spacing]
    matPos[1,brkpt_spacing:Sfinal-brkpt_spacing+1] = [i for i in range(brkpt_spacing,Sfinal-brkpt_spacing+1)]
    #for nbs in range(2,numBrks+1):
    for nbs in range(2,3):
        # matCost(nbs, beginIdx) to matCost(nbs,EndIdx) will get filled
        beginIdx = nbs * brkpt_spacing          #for nbs=2, h=10, beginIdx=20
        endIdx = Sfinal - brkpt_spacing       # endIdx = n-10 = 100-90 = 90 (say)
#        print "beginIdx, endIdx=",  beginIdx, endIdx
        # vecCost will get filled from potIdxBegin to potIdnEnd.
        # matCost at nbs point will be based on minimizing vecCost[potIdxBegin:potIdxEnd]
        potIdxBegin = (nbs-1) * brkpt_spacing   #for nbs=2,h=10, potIdxBegin=10
        for idx in range(beginIdx, endIdx+1):
            potIdxEnd = min(idx, Sfinal-brkpt_spacing)
            vecCost = [99999 for i in range(Sfinal+1)]
            for j in range(potIdxBegin, potIdxEnd+1):
                #RSSTri_full(j+1, idx) <-- cost of ([j+1, idx]), cost of [idx+1, n) is added later (see note 1)
                #matCost(nbs-1, j) <-- cost of [1:j] using nbs-1 breakpoints
                #TODO: are we supposed to take matCost[nbs-1,j] or matCost[1,j]  ?
                vecCost[j] = matCost[nbs-1, j] + RSSTri_full[j][idx-1]  # remember that RSSTri_full is stored in the regular python way
#                print "Values read: ", j, ",", idx-1
                                                                      # matCost is stored in python style with 0 row 0 col empty
            
            #cost of nbs split (last one at idx)
            matCost[nbs, idx]  = min(vecCost)
            matPos[nbs, idx] = np.argmin(vecCost) #argmin returns the first one which is what we want to use
    
    #note 1:  add the cost of  segment(idx+1, n)
    for idx in range(numBrks * brkpt_spacing, Sfinal-brkpt_spacing):
        matCost[numBrks, idx] = matCost[numBrks, idx] + RSSTri_full[idx][Sfinal-1]
        
    tmp = np.argmin(matCost[numBrks, 0: Sfinal-brkpt_spacing + 1])
    
    last_brkpt_pos = tmp    #   + numBrks * brkpt_spacing
    curr_brkpt_pos = last_brkpt_pos 
    vecBrkPts = [-1 for i in range(0, numBrks)]
    vecBrkPts[numBrks-1] = last_brkpt_pos
    i = numBrks
    while(i > 1):
        curr_brkpt_pos = matPos[i, curr_brkpt_pos]
#        print i, curr_brkpt_pos
        i = i -1
        vecBrkPts[i-1] = curr_brkpt_pos + 1  #we are ignoring the 0-row,0-column
    
    #shift everything down by 1 to match the original vec_timestamps indices ... remember they start from 0.
    vecBrkPts = [int(j-1) for j in vecBrkPts]
    try:
        index0 = vecBrkPts.index(0)
        del (vecBrkPts[index0])
    except:
        #do nothing
        print "no 0 in vecBrkPts"
        
    try:
        indexSfinal = vecBrkPts(Sfinal-1)
        del (vecBrkPts[indexSfinal])
    except:
        junk = 1
#        print "no Sfinal in vecBrkPts"n

    return vecBrkPts

    
def hammingDist(list1, list2):
    
    len_list1 = len(list1)
    len_list2 = len(list2)
    
    if (len_list1 == 0) and (len_list2 == 0):
        return 0
    elif len_list1 == 0:
        return len_list2
    elif len_list2 == 0:
        return len_list2
        
    if len_list1 > len_list2 :
        list1 = list1[0:len_list2]
        
    if len_list2 > len_list1:
        list2 = list2[0:len_list1]
        
    return sum(c1 != c2 for c1, c2 in zip(list1, list2)) + abs(len_list2 - len_list1)


def bfast(tyeardoy, vec_obs_all, \
          ewma_trainingStart, ewma_trainingEnd, ewma_lowthreshold, ewma_K, \
          frequency, numBrks, harmonicDeg, h, numColsProcess, pval_thresh, maxIter):

    num_obs = len(vec_obs_all)

    # ************* develop the presInd vector ***********************
    presInd = np.where(vec_obs_all > ewma_lowthreshold)[0]
    tyeardoy_idxs = np.where(np.logical_and(ewma_trainingStart<= tyeardoy[:,0], \
                                            tyeardoy[:,0]< ewma_trainingEnd))[0]
    common_idx = list(set(tyeardoy_idxs).intersection(presInd))
    training_t = tyeardoy[common_idx, 1]

    #Corner case
    if (len(training_t) < 2 * ewma_K + 1):    #from ewmacd
        brkPtsGlobalIndex = [0, num_obs-1]
        brkPtYrDoy = [tyeardoy[i,:] for i in brkPtsGlobalIndex] 
        vecTrendFitFull = [-2222]*num_obs
        brkpt_summary = [-2222]*num_obs
        return brkPtsGlobalIndex, brkPtYrDoy, vecTrendFitFull, brkpt_summary
                
    ind = 0
    num_days_gone = 0
    vec_timestamps_edited = np.zeros(num_obs)
    vec_timestamps_edited[0] = tyeardoy[ind, 1]
    while ind < num_obs:
        if ind > 0:
            if (tyeardoy[ind,0] != tyeardoy[ind-1, 0]):
                if (tyeardoy[ind-1, 0] % 4) == 0:
                    num_days_gone = num_days_gone + 366
                else:
                    num_days_gone = num_days_gone + 365
                    
                vec_timestamps_edited[ind] = num_days_gone + tyeardoy[ind, 1]
            else:
                vec_timestamps_edited[ind] = num_days_gone + tyeardoy[ind, 1]
                        
        ind += 1

    #*************** prepare data ***********************************
    vec_obs = vec_obs_all[presInd]  #    vec_obs_pres or D
    vec_timestamps_edited_pres = vec_timestamps_edited[presInd]  
    vec_timestamps_pres =  tyeardoy[presInd, 1]   # t
    vec_timestamps_pres_harmonic = map(lambda x: x * 2 * np.pi/365.0,  vec_timestamps_pres)
    Sfinal = len(vec_timestamps_pres)  # length of present data   
    
    #*********** actual processing starts here *******************
    
    #initialize
    vecSeasonFit = np.zeros(Sfinal)  
    vecTrendFit = np.zeros(Sfinal)
    vecTrendBrks = np.zeros(numBrks+2)  
    vecSeasonalBrks = np.zeros(numBrks+2) 
    vecTrendBrksOld = np.ones(numBrks+2)  
    vecSeasonalBrksOld  = np.ones(numBrks+2)
    hamTrend = hammingDist(vecTrendBrks, vecTrendBrksOld)
    hamSeason = hammingDist(vecSeasonalBrks, vecSeasonalBrksOld)

#    np = frequency   #number of times the season appears in one cycle of the ts. Figure out how to determine this in general.
#    ns = 10 * np + 1
#    nt = np.ceiling(1.5 * np)
#    nl = 23
#    isdeg = 0
#    itdeg = 1
#    ildeg = 1
#    nsjump = Sfinal    # ceiling(nt/10)
#    ntjump = 4      # ceiling(ns/10)
#    nljump = 3      # ceiling(nl/10)
#    ni = 2
#    no = 0
#    rw = 1  #todo
#    work = 1 # todo
#    n = len(y)  #todo
#    st.stl(vec_obs, len(vec_obs), \
#            np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni,no, \
#            weights, vec_fit, vec_u, work)
    
    it = 0
    
    while ( (hamTrend != 0) or (hamSeason != 0)) and (it < maxIter):

        vecTrendBrksOld = copy.deepcopy(vecTrendBrks)
        vecSeasonalBrksOld = copy.deepcopy(vecSeasonalBrks)
        
        # "adjust" the data by deseasoning
        u = vec_obs - vecSeasonFit  #[vec_obs[i] - vecSeasonFit[i]  for i in range(0,Sfinal)]

        # get OLS-MOSUM statistics
        process = OLSMosum(vec_timestamps_edited_pres, u, "linear", h, 0)
        pValTrend = calcPValue(np.abs(max(process)), "brownian bridge increments", numColsProcess, h, "max")
#        print pValTrend, pval_thresh, h
        # get breakpoints in trend
#        print 'it = ', it, ', pValTrend =', pValTrend, ', pval_thresh =', pval_thresh
        if pValTrend <= pval_thresh:
            RSStri = getRSStri(vec_timestamps_edited_pres, u, "linear", h, 0)
            a = buildDynPrTable(RSStri, numBrks, Sfinal, h)
            vecTrendBrks = np.concatenate(([0], a))
            vecTrendBrks = np.concatenate((vecTrendBrks, [Sfinal-1]))    
#            print 'vecTrendBrks:', vecTrendBrks
            # do piecewise linear approximation
            numInternalBrksFinal = len(vecTrendBrks) - 2
            numTrendSegs = numInternalBrksFinal + 1
            vecTrendFit = [0] * Sfinal 
            linCoefs = [[0,0]]*numTrendSegs 
            for i in range(0, numTrendSegs):
                startPoint = vecTrendBrks[i]
                endPoint = vecTrendBrks[i+1]
                if (i == numTrendSegs-1):   #last segment
                    x = vec_timestamps_edited_pres[startPoint : endPoint+1]
                    y = u[startPoint : endPoint+1]
                    linCoefs[i], vecTrendFit[startPoint:endPoint+1] = regress(x, y, "linear", 0)
                else:
                    x = vec_timestamps_edited_pres[startPoint : endPoint]
                    y = u[startPoint : endPoint]
                    linCoefs[i], vecTrendFit[startPoint:endPoint] = regress(x, y, "linear", 0)
        else:
            vecTrendBrks = [0, Sfinal-1]
            numTrendSegs = 1
            vecTrendFit = [0] * Sfinal
            linCoefs = [[0,0]]*numTrendSegs
            linCoefs[0], vecTrendFit[0:Sfinal] = regress(vec_timestamps_edited_pres, u, "linear", 0)

#######_____________________________________________________________________________________######
#######_____________________________________________________________________________________######

        # adjust the data by detrending (use the most recently  calculated, i.e., updated, trend model)
        util = vec_obs - vecTrendFit

        # get OLS-MOSUM statistics for seasonal
        process = OLSMosum(vec_timestamps_pres_harmonic, util, "harmon", h, harmonicDeg)
        pValSeason = calcPValue(np.abs(max(process)), "brownian bridge increments", numColsProcess, h, "max")
        
        # get breakpoints in season
        if pValSeason <= pval_thresh:
            RSStri = getRSStri(vec_timestamps_pres_harmonic, util, "harmonic", h, harmonicDeg)
            a = buildDynPrTable(RSStri, numBrks, Sfinal, h)   #gives us the indices of the brkpt locations.
            vecSeasonalBrks = np.concatenate(([0], a))
            vecSeasonalBrks = np.concatenate((vecSeasonalBrks, [Sfinal-1]))
            
            # do piecewise harmonic approximation
            numInternalBrksFinal = len(vecSeasonalBrks) - 2
            numSeasonSegs = numInternalBrksFinal + 1
            vecSeasonFit = [0] * Sfinal  
            harmCoefs = [[0,0]] * numSeasonSegs
            for i in range(numSeasonSegs):
                startPoint = vecSeasonalBrks[i]
                endPoint = vecSeasonalBrks[i+1]
                if (i == numSeasonSegs-1):   #last segment
                    x = vec_timestamps_pres_harmonic[startPoint : endPoint+1]
                    y = util[startPoint : endPoint+1]
                    harmCoefs[i], vecSeasonFit[startPoint:endPoint+1] = regress(x, y, "harmon", harmonicDeg)
                else:
                    x = vec_timestamps_pres_harmonic[startPoint : endPoint]
                    y = util[startPoint : endPoint]
                    harmCoefs[i], vecSeasonFit[startPoint:endPoint] = regress(x, y, "harmon", harmonicDeg)
        else:
            vecSeasonalBrks = [0, Sfinal-1]
            numSeasonSegs = 1
            vecSeasonFit = [0] * Sfinal
            harmCoefs = [[0,0]] * numSeasonSegs
            harmCoefs[0], vecSeasonFit[0:Sfinal] = regress(vec_timestamps_pres_harmonic, util, "harmon", harmonicDeg)

        # get the Hamming distance between the previous brkpts and current brkpts.
        hamTrend = hammingDist(vecTrendBrks, vecTrendBrksOld)
        hamSeason = hammingDist(vecSeasonalBrks, vecSeasonalBrksOld)
        it += 1
        
    # merge/drop unnecessary brkpts. 
    #Most updated set of trend breaks is in vecTrendBrks
    #Most recent construction is in vecTrendFit
#    vecTrendBrks_final = [vecTrendBrks[0]]
#    for i in vecTrendBrks[1:-1]:
#        this_seg_endpt_val = 1
#        next_seg_startpt_val = 1 
#        if abs(this_seg_endpt_val - next_seg_startpt_val) > brkpt_mag_thresh:
#            vecTrendBrks_final.append(i)
#        
#    vecTrendBrks_final.append(vecTrendBrks[-1])
    
    # get reconstruction on original (all) timepoints
    vecTrendFitFull = np.zeros(num_obs)
    brkPtsGlobalIndex = presInd[vecTrendBrks]
    brkPtsGlobalIndex[0] = 0           #to account for the case when the very first obs is missing
    brkPtsGlobalIndex[-1] = num_obs-1  #replace the last 'present' index with last actual index
    final_numTrendSegs = len(vecTrendBrks) - 1
    for i in range(final_numTrendSegs):
        startPoint = brkPtsGlobalIndex[i]
        endPoint   = brkPtsGlobalIndex[i+1]
        if (i == final_numTrendSegs-1):
            #use present indices to get fit. This is the most recently computed linear coeffs.
            #project on ALL indices
            vecTrendFitFull[startPoint:endPoint+1] = linCoefs[i][0] + linCoefs[i][1]*vec_timestamps_edited[startPoint:endPoint+1]
        else:
            # use present indices to get fit. project on ALL indices
            vecTrendFitFull[startPoint:endPoint] = linCoefs[i][0] + linCoefs[i][1]*vec_timestamps_edited[startPoint:endPoint]
            
    brkPtYrDoy = [tyeardoy[i,:] for i in brkPtsGlobalIndex]
    brkpt_summary = [0 for i in range(num_obs)]
    for i in brkPtsGlobalIndex[1:-1]:
        brkpt_summary[i] = vecTrendFitFull[i] - vecTrendFitFull[i-1]

    return brkPtsGlobalIndex, brkPtYrDoy, vecTrendFitFull, brkpt_summary
