#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:16:41 2018

@author: rishu
"""

time1 = 2000
time2 = 2012 + 1
eps = 1
if time1 == 2000:
    tickGap = 2
    dist_thresh = 13.0  # because an otherwise good algorithm outcome could have one breakpoint lurking around at far end.
                        # We don't want to lose this result just because of that false alarm.
    min_ew_brks = 0
    max_brks = 3
else:
    tickGap = 4
    dist_thresh = 25.0  # because an otherwise good algorithm outcome could have one breakpoint lurking around at far end.
                        # We don't want to lose this result just because of that false alarm.
    min_ew_brks = 1
    max_brks = 5     # for 13 year time period, 3 breaks would be enough?!
                     # For 25 years, 5 break would be good. 
                     # For 6 years, 2 breaks would be enough.

maskVal = -9999
# parameters for ewmacd
ew_num_harmonics = 2  # was 2 earlier
ew_ns = ew_num_harmonics
ew_nc = ew_num_harmonics
ew_xbarlimit1 = 1.5
ew_xbarlimit2 = 20
ew_lowthreshold = 0
ew_trainingStart = time1
if time1 == 2000:
    ew_trainingEnd= time1 + 2
else:
    ew_trainingEnd= time1 + 3
ew_mu = 0
ew_L = 3.0   # default is 3.0
ew_lam = 0.5   # default is 0.5
ew_persistence = 7  # default is 7
ew_summaryMethod = 'on-the-fly'   #'reduced_wiggles'  # 'annual_mean'  #

# parameters for bfast
bf_h = 0.15
bf_numBrks = 2 #max_brks
bf_numColsProcess = 1
bf_num_harmonics = ew_num_harmonics  #1
bf_pval_thresh = 0.05  #default is 0.05
bf_maxIter = 2
bf_frequency = 23

# parameters for landtrendr
ltr_despike_tol = 0.9  # 1.0, 0.9, 0.75, default is 0.9
ltr_pval = 0.2          # 0.05, 0.1, 0.2, default is 0.2
ltr_mu = 6  #max_brks + 1   #mpnu1     # 4, 5, 6, default is 6
ltr_recovery_threshold = 1.0  # 1, 0.5, 0 25
ltr_nu = 3              # 0, 3
ltr_distwtfactor = 2.0   #i have taken this value from the IDL code
ltr_use_fstat = 0        #0 means 'use p_of_f', '1' is for 'dont use p_of_f'.
                               #So if use_fstat = 0, the code will use p_of_f.

ltr_best_model_proportion = 0.75


# parameters for plots
colors = ['green', 'sandybrown', 'black']
dashes = [[12, 6, 12, 6], [12, 6, 3, 6], [3, 3, 3, 3]] # 10 points on, 5 off, 100 on, 5 off
#    line_styles = ['--','s--',':']
line_widths = [4, 5, 2]
