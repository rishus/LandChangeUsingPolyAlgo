#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:01:50 2018

@author: rishu
"""

def regress(t, u, model, K):
    
    M = len(t)
    if (model == 'linear') or (M < 2*K+1):
        ncols = 2
    elif (model == "harmon"):
        ncols = 2*K+1
    else:
        print ('model not supplied')

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
        print ("model not supported")
    
    if (np.abs(np.linalg.det(np.dot(np.transpose(X), X))) < 0.001):
        alpha_star = [0,0]
        fit = np.zeros((M,))
        return alpha_star, fit

    alpha = np.linalg.solve(np.dot(np.transpose(X), X), np.dot(np.transpose(X), u))
    fit = np.dot(X,alpha)
#    print 'fit_dims: ', fit.shape

    return alpha, fit
