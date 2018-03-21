#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 08:26:14 2016

@author: rishu
"""

import numpy as np
import bfast  as bf
import ewmacd as ew
import landTrendR  as ltr
# for remote desktop:
#import matplotlib as mpl
#mpl.use('Agg')
from collections import defaultdict
from matplotlib import pylab as plt
import datetime as dt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import parameters 

####### read data from timesync dataset #####################
def data_from_timeSyncCsvFile(path, mat_all_lines, fn_groundTruth, my_pid, time_start, time_end):
    """
    Inputs:
        (i) my_pid: the id of the pixel currently being investigated. Each
                    pixel has a unique pixel-id.
    
    This function reads the Timesync data files for pixe with id my_pid. That data has
    (i)  the spectral time series for each pixel (multiple bands)
    (ii) information on land use changes (year+doy and nature) for each pixel.
    
    The time series and the land use change info are stored in two separate files.
    
    Only visual red (R) and NIR bands are used currently.
    This function reads the R and NIR time series, and generates the NDVI time series.
    
    An integer -9999 is used as masking value.
    
    Outputs:
        (i)    vec_obs:   1D array of dimension 1 x num_obs.
                          Contains the NDVI time series for pixel with id my_pid.
                        
        (ii)   tyeardoy:  2D array if dimension num_obs x 2. 
                          Each row contains the year and the doy of the corresponding observation
                          in vec_obs.
        
        (iii)  changes:   A list of length num_changes + 1, where num_changes is the number of changes.
                          The zeroth element is simply the number num_changes.
                          Each of the first to num_changes element is a 5-element list of the form
                          [starting year, ending year, change_type, prechange land cover, postchange land cover]
                          
                          For a stable pixel, num_obs = 0.
    
    """

    num_lines = len(mat_all_lines)
    this_pixel_info ={'sensor': [], 'pid':  [], 'yr': [],  \
                      'doy': [], 'b3': [], 'b4': [], 'b5': [], \
                      'b6': []}
    num_obs = 0
    for line in range(num_lines):
        mat_vals = mat_all_lines[line].strip().split(',')
        if (int(mat_vals[1]) == my_pid) and (mat_vals[0]!= '' and \
                      mat_vals[1]!= '' and mat_vals[2] != '' and \
                      mat_vals[4]!= '' and mat_vals[5]!='' and \
                      mat_vals[8]!='' and mat_vals[9]!='' and 
                      mat_vals[10]!='' and mat_vals[12]!=''): # and \
#                      int(mat_vals[13])==0):
            #sensor, pid, tsa, plotid, year, doy, band3, band4, band5, band6
                
            this_pixel_info['sensor'].append(int(mat_vals[0][2]))
            this_pixel_info['yr'].append(int(mat_vals[4]))
            this_pixel_info['doy'].append(int(mat_vals[5]))
            try:
                if (int(mat_vals[13]) not in [0, 1]):   # cloud, water etc masking
                    this_pixel_info['b3'].append(maskVal)
                    this_pixel_info['b4'].append(maskVal)
                    this_pixel_info['b5'].append(maskVal)
                    this_pixel_info['b6'].append(maskVal)
                else:
                    this_pixel_info['b3'].append(int(mat_vals[8]))
                    this_pixel_info['b4'].append(int(mat_vals[9]))
                    this_pixel_info['b5'].append(int(mat_vals[10]))
                    this_pixel_info['b6'].append(int(mat_vals[12]))
            except:
                    this_pixel_info['b3'].append(maskVal)
                    this_pixel_info['b4'].append(maskVal)
                    this_pixel_info['b5'].append(maskVal)
                    this_pixel_info['b6'].append(maskVal)
            num_obs +=1

    tyeardoy_all = np.zeros((num_obs, 2))
    vec_obs_all = []
    tyeardoy_all[:, 0] = this_pixel_info['yr']
    tyeardoy_all[:, 1] = this_pixel_info['doy']
    for i in range(num_obs):
        red = float(this_pixel_info['b3'][i])
        nir = float(this_pixel_info['b4'][i])
        if (abs(nir+red) < np.finfo(float).eps):
            vec_obs_all.append(maskVal)
        else:
            ndvi = ((nir-red)/(nir+red))
            if ndvi < 0 or ndvi >1:
                vec_obs_all.append(maskVal)
            else:
                vec_obs_all.append(ndvi * 10000)

    # limit returns to the desired time span
    a = [i for i in range(len(vec_obs_all)) \
            if (tyeardoy_all[i, 0] >= time_start) and (tyeardoy_all[i, 0] < time_end)]

    tyeardoy = np.zeros((len(a), 2))
    ctr = 0
    for i in a:
        tyeardoy[ctr, 0] = int(tyeardoy_all[i, 0])
        tyeardoy[ctr, 1] = int(tyeardoy_all[i, 1])
        ctr += 1

    vec_obs = np.asarray([vec_obs_all[i] for i in a])
    # notice that tyeardoy is sent out as an array while vec_obs is sent as a list.
    # well, not any more
    
    # Now get the ground truths. Note that only disturbance pixels are included in the
    # ground truth sheet.
    mat_all_changes = []
    with open(fn_groundTruth, 'r') as f_gt:
        first_line = f_gt.readline()
        for i, line in enumerate(f_gt):
            this_line = line.strip().split(',')
            if int(this_line[0]) == my_pid:
                mat_all_changes.append(line)
            pass

    num_changes = 0
    changes = []
    # Note that if pid is a no-disturbance-ever pixel, then mat_all_changes will be empty anyways.
    # So this loop won't run. 
    # So, basically, the 'else' statement in the loop gets executed only for pids where some lu-cover info is available.
    for line in mat_all_changes:
        mat_gt_vals = line.strip().split(',')
        s_yr = mat_gt_vals[1]
        e_yr = mat_gt_vals[2]
        s_lu = mat_gt_vals[5]
        e_lu = mat_gt_vals[6]
        if ((int(s_yr) >= time_start) and (int(e_yr) <= time_end)):
            num_changes +=1
            change_type = mat_gt_vals[3]
            changes.append([s_yr, e_yr, change_type, s_lu, e_lu])
        else:
            changes.append(['x','x','x', s_lu, e_lu])
    changes = [num_changes] + changes

    return vec_obs, tyeardoy, changes


def dist(A, B, toprint='no'):
    """
    Calculates the directional Hausdorff distance between two arrays
    """

    lA = len(A)
    lB = len(B)
    if lA == 0 and lB == 0:
        return 0
        
    if lA == 0 and lB != 0:
        return 1000000
    if lB == 0 and lA != 0:
        return 1000000

    dists_AB = np.zeros((lA, lB))
    daB = np.zeros( (lA,) )
    for i in range(lA):
        # get d(a, B)
        for j in range(lB):
            dists_AB[i, j] = abs(A[i]-B[j])
            
        daB[i] = min(dists_AB[i,:])
#    if toprint == 'bl':
#        print 'BFAST, LTR:'
#    if toprint == 'lb':
#        print 'LTR, BFAST'
#    if toprint in ['bl','lb']:
#        print 'A: ', A
#        print 'B: ', B
#        for i in range(lA):
#            print dists_AB[i,:]
#        
#        print 'daB = ', daB

    dAB = max(daB)
    
    return dAB


################# actual polyalgorithm is here ######################    
def process_pixel(tyeardoy, vec_obs_original, pixel_info, tickGap, changes, dist_thresh):
    """
    Inputs: 
         (i)  tyeardoy:           2D array if dimension num_obs x 2. 
                                  Each row contains the year and the doy of the corresponding observation
                                  in vec_obs.
                                  
        (ii)  vec_obs_original:   List of timeseries, each timeseries being a 1D vector.
                                  Each timeseries corresponds to one spectral (raw or derived) band.
                                  If multiple bands are being considered, then there will
                                  be multiple timeseries.
                                  
        (iii) changes:            changes:   A list of length num_changes + 1, 
                                  where num_changes is the number of changes.
                                  
        (iv)  dist_thresh :       Is already available. Doesn't quite need to be passed.
        
        
    This code calls each algorithm, collects their results, decides the 'final' algorithm for the current pixel,
    and returns the breakpoints of the individual algos as well as the polyalgo.
    Currently has redundancies but will improve as the polyalgorithm methodology evolves.
            
    Outputs:
          
        Breakpoints of each algorithm
        (i)   bf_brkpts
        (ii)  ew_brkpts
        (iii) ltr_brkpts
        (iv)  polyAlgo_brkpts
        
        Each of the above variables is a list of the form
                  [b1 = [year, doy], b2 = [year, doy], ..., bp = [year, doy]],
        where bi corresponds to the i-th breakpoint. If an algorithm does not find any
        breakpoint, then this list is empty ([]).

        The name of the algorithm that got selected is also output (winner)
        but this is only for gathering code statistics. It is not really needed
        and will be removed evenutally.
                  

    """
# this subroutine is for a fixed set of parameters. It is able to process multiple bands

    time1 = pixel_info[0]
    pid = pixel_info[1]
    num_bands = len(vec_obs_original)  # becuz vec_obs_original is a list
    num_obs = tyeardoy.shape[0]

    for band in range(num_bands):

        # ************* develop the presInd vector and check training data availability ***********************
        presInd = np.where(vec_obs_original[band] > ew_lowthreshold)[0]      # all presentIds.
        tyeardoy_idxs = np.where(np.logical_and(ew_trainingStart<= tyeardoy[:,0], \
                                                tyeardoy[:,0]< ew_trainingEnd))[0] # all indices in the training period.
        common_idx = list(set(tyeardoy_idxs).intersection(presInd))  # presentIds in the training period.
        training_t = tyeardoy[common_idx, 1]    # only doys in the training period.
        #Corner case
        if (len(training_t) < 2 * ew_num_harmonics + 1):    #from ewmacd
           print   (pid, ': too little training data')
           return [], [], [], [], 'insuff'

#        brkptsummary is needed in these 1D codes to make 1D plots. But for 2D, it is redundant.
        tmp2, ewma_summary, ew_brks_GI, ew_brkpts, ew_brkptsummary =    \
                                 ew.ewmacd(tyeardoy, vec_obs_original[band], presInd, \
                                 ew_num_harmonics, ew_xbarlimit1, ew_xbarlimit2, \
                                 ew_lowthreshold, ew_trainingStart, ew_trainingEnd, \
                                 ew_mu, ew_L, ew_lam, ew_persistence, \
                                 ew_summaryMethod, ew_ns, ew_nc, 'dummy')

        bf_brks_GI, bf_brkpts, bfast_trendFit, bf_brkptsummary = \
                    bf.bfast(tyeardoy, vec_obs_original[band], presInd, \
                             ew_trainingStart, ew_trainingEnd, ew_lowthreshold, ew_num_harmonics, \
                             bf_frequency, bf_numBrks, bf_num_harmonics, bf_h, bf_numColsProcess, \
                             bf_pval_thresh, bf_maxIter)

        bestModelInd, allmodels_LandTrend, ltr_brks_GI, ltr_brkpts, ltr_trendFit, ltr_brkptsummary = \
                        ltr.landTrend(tyeardoy, vec_obs_original[band], presInd, \
                          ew_trainingStart, ew_trainingEnd, ew_lowthreshold, ew_num_harmonics, \
                          ltr_despike_tol, ltr_mu, ltr_nu, ltr_distwtfactor, \
                          ltr_recovery_threshold, ltr_use_fstat, ltr_best_model_proportion, \
                          ltr_pval)

        ####### compare different algorithm results #########
        # pairwise distance between brkpts
        # Does any algorithm indicate breakpoint during the training period?
        use_ewma = 'yes'
        for brk in bf_brkpts[1:]:
            if brk[0] in range(ew_trainingStart, ew_trainingEnd+1):
                use_ewma = 'no'

        bf_brkpts_m = [(i[0] + min(i[1], 365)/365.) for i in bf_brkpts[1:-1]]
        ltr_brkpts_m = [(i[0] + min(i[1], 365)/365.) for i in ltr_brkpts[1:-1]]
        ew_brkpts_m = [(i[0] + min(i[1], 365)/365.) for i in ew_brkpts[1:-1]]
        dist_BL = dist(bf_brkpts_m, ltr_brkpts_m, 'no')
        dist_LB = dist(ltr_brkpts_m, bf_brkpts_m, 'no')
        if use_ewma == 'yes':
            dist_BE = dist(bf_brkpts_m, ew_brkpts_m,'no')
            dist_EB = dist(ew_brkpts_m, bf_brkpts_m,'no')
            dist_LE = dist(ltr_brkpts_m, ew_brkpts_m, 'no')
            dist_EL = dist(ew_brkpts_m, ltr_brkpts_m, 'no')
        else:
            dist_BE = 1000000
            dist_EB = 1000000
            dist_LE = 1000000
            dist_EL = 1000000
        
        vec_dists = [dist_BE, dist_EB, dist_BL, dist_LB, dist_LE, dist_EL]
        s_ind = vec_dists.index(min(vec_dists))
        if vec_dists[s_ind] <= dist_thresh:
            if s_ind in [0, 2]:
                polyAlgo_brkpts = bf_brkpts
                winner = 'bf'
            elif s_ind in [1, 5]:
                polyAlgo_brkpts = ew_brkpts
                winner = 'ew'
            else:  # s_ind in [3, 4]
                polyAlgo_brkpts = ltr_brkpts
                winner = 'ltr'
        else:
            # if no agreement, then declare it stable
            polyAlgo_brkpts = [0, num_obs-1]
            winner = 'none'
        
        with open("polyalgo_distances.csv", "a") as fh:
            fh.write(str(pid) + ', ' + str(vec_dists[s_ind]) + ', ' +  str(dist_BE) + ',   ' + str(dist_EB) +  ',   '  +  \
                                                           str(dist_BL) + ',   ' + str(dist_LB) +  ',   '   + \
                                                           str(dist_LE) + ',   ' + str(dist_EL) +  ',   '  + \
                                                           str(s_ind) + ',    ' + winner + '\n')
        fh.close()
        
        ####### plot outputs and input #########
        if len(presInd) == 0:
            print 'presInd = 0 in processPixel'
            return

        bf_trendFit_scaled = [float(bfast_trendFit[i])/float(10000) for i in range(num_obs)]
        bf_trendFit_scaled = [float(bf_brkptsummary[i])/float(10000) for i in range(num_obs)]
#        ltr_trendFit_scaled = [float(ltr_trendFit[i])/float(10000) for i in range(num_obs)]
        ltr_trendFit_scaled = [float(ltr_brkptsummary[i])/10.0 for i in range(num_obs)]
#        ew_flags_scaled = [float(ewma_summary[i])/float(10) for i in range(num_obs)]
        ew_flags_scaled = [float(ew_brkptsummary[i])/float(10) for i in range(num_obs)]
        
        plot_trajectories(pid, time1, tyeardoy, num_obs, vec_obs_original[band], presInd, \
                      tickGap, ew_flags_scaled, ltr_trendFit_scaled, bf_trendFit_scaled, \
                      winner, changes, colors, dashes, line_widths)

    return bf_brkpts[1:-1], ew_brkpts[1:-1], ltr_brkpts[1:-1], polyAlgo_brkpts[1:-1], winner


def plot_trajectories(pid, time1, tyeardoy, num_obs, vec_obs_original, presInd, \
                      tickGap, ew_flags_scaled, ltr_trend_scaled1, bf_trendFit_scaled, \
                      winner, changes, colors, dashes, line_widths):
    """
    script for generating the 1D plots included in the thesis and the ISPRS paper.
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    ax.plot()
    years_all = [int(tyeardoy[i, 0]) for i in range(0, num_obs)]
    doys_all = [int(tyeardoy[i, 1]) for i in range(0, num_obs)]
    dates_all = [dt.datetime(year, 1, 1) + dt.timedelta(day - 1) for year,day in zip(years_all,doys_all)]
    vec_obs_pres = [vec_obs_original[i] for i in presInd]
    Sfinal = len(presInd)
    years = [int(tyeardoy[i, 0]) for i in presInd]
    doys = [int(tyeardoy[i, 1]) for i in presInd]
    dates = [dt.datetime(year, 1, 1) + dt.timedelta(day - 1) for year,day in zip(years,doys)]
    datemin = dt.date(min(dates).year, 1, 1)
    datemax = dt.date(max(dates).year + 1, 1, 1)
    year_tics = sorted(set([dt.date(d.year,1, 1) for d in dates]))
    ticklabels = ['']*len(year_tics)
    ticklabels[::tickGap] = [item.strftime('%Y') for item in year_tics[::tickGap]]    # Every 5th ticklable shows the month and day

    vec_obs_pres_plot = [float(vec_obs_pres[i])/float(10000) for i in range(Sfinal)]
    line, = ax.plot(dates, vec_obs_pres_plot, '--', color=colors[0], \
            linewidth=2, markersize=6 , label='NDVI')  #+str(band))
    line.set_dashes(dashes[0])
    ax.plot(dates_all, ltr_trend_scaled1, '-', color=colors[0], linewidth=line_widths[0], label='LandTrendR')
    ax.plot(dates_all, bf_trendFit_scaled, '-', color=colors[1], linewidth=line_widths[0], label='BFAST')
    ax.plot(dates_all, ew_flags_scaled, '--', color=colors[2], linewidth=3, label='EWMACD/10')

    legend = ax.legend(loc='best', fontsize=14, shadow=True)
    frame = legend.get_frame()
    legend.get_frame().set_alpha(0.5)
    frame.set_facecolor('white')
    years = mdates.YearLocator(tickGap, month=7, day=4)   # Tick every 5 years on Jan 1st #July 4th
    yearsFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.set_xlim(datemin, datemax)
#        ax.set_ylim(-0.1, 1.0)  #(min(ewmacd_flags_scaled)-0.05, 1.0)
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels[::tickGap]))
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    title_str = []
    xlabel_str = []
#        changes = [s_yr, e_yr, change_type, s_lu, e_lu]
    for element in changes[1:]:
        tmp = ','.join([element[0], element[1], element[2][0:3]])
        tmp_lu_changes = '-->.'.join([element[3][0:3], element[4][0:3]])
        title_str.append(tmp)
        xlabel_str.append(tmp_lu_changes)
    plt.title(title_str)
    xlabel_str.append('  ' + winner)
    plt.xlabel(xlabel_str)
    plt.rcParams.update({'font.size': 22})

    fig_path = ''
    
    fig_name = 'all_' + str(time1) + "to12_" + str(pid) + '_nobias'+  '.png'
#    fig_name =  'ltr_' + str(time1) + "to12_" + str(pid) + \
#                      '_despike_tol' + str(ltr_despike_tol) + \
#                     ',pval' + str(ltr_pval) + ',mu' + str(ltr_mu) + \
#                      ',nu' + str(ltr_nu) + ',trec' + str(ltr_recovery_threshold) + '.eps'
#    fig_name =  'ew_' + str(time1) + "to12_" + str(pid) +  \
#                     ',lam' + str(ew_lam) + ',L' + str(ew_L) + \
#                     ',pers' + str(ew_persistence) + '.eps'
#    fig_name =  'bf_' + str(time1) + "to12_" + str(pid) +  '.eps'  #\
#                     '_h' + str(bf_h) + ',brks' + str(bf_numBrks) + \
#                     ',pval' + str(bf_pval_thresh) + ',numHarm' + str(bf_num_harmonics) + '.eps'
    full_fig_name = fig_path  + fig_name
    fig.savefig(full_fig_name, bbox_inches='tight', format='png')
    
    return


def process_timesync_pixels(path = ""):
    """
    main routine
    (i)   input file names are provided here.
    (ii)  for each pixel, the time series data and the 'reference' data is read
          using a function call.
    (iii) a for loop runs over the pixels.
    (iii) for each pixel, the breakpoints are calculated by calling the function process_pixel.
    (iv)  process_pixel outputs the breakpoints for each algorithm 
          as a list 
          
          [b1 = [year, doy], b2 = [year, doy], ..., bp = [year, doy]],
          
          where each bi is a breakpoint found by the algorithm.
    (iv)  in this code, the output is not saved as such. Saving the outputs
          involves only minor additional scripting. It is implementd in the 2D
          version.
    (v)   Visualization of breakpoints is also implemented in the 2D version.
    
    """

    fn_timeSync_pids = path + "ts_pids_list.csv"
    fn_timeSync_ts = path + 'conus_spectrals.csv'
    fn_timeSync_disturbance = path + "ts_disturbance_segments.csv'  #

    pixels_list = []
    with open(fn_timeSync_pids, 'r') as f:
        for i, line in enumerate(f):
            line_vals = line.strip().split(',')
            pixel = int(line_vals[0])
            if pixel not in pixels_list:
                pixels_list.append(pixel)
            pass

    mat_all_timeseries = defaultdict(list)
    with open(fn_timeSync_ts, 'r') as f:
        first_line = f.readline()
        for i, line in enumerate(f):
            this_line = line.strip().split(',')
            if int(this_line[1]) in pixels_list:
                mat_all_timeseries[int(this_line[1])].append(line)
            pass

    problem_pixels = []
#    print 'total pixels =', len(pixels_list)
    fn_pids = []
    for pixel in pixels_list:  #40027038, 38029024]:  #pixels_list:

        my_pid = pixel  #int(pixel[0:-1])
#        print my_pid
        #######################################################################
        vec_obs_original_ndvi, tyeardoy, changes = data_from_timeSyncCsvFile( \
                            path, mat_all_timeseries[my_pid], fn_timeSync_disturbance, my_pid, time1, time2)
        
        if (len(tyeardoy) != len(vec_obs_original_ndvi)) or (len(tyeardoy) == 0):
            problem_pixels.append(my_pid)
            continue

        vec_obs_original = [] 
        vec_obs_original.append(vec_obs_original_ndvi)
        # vec_obs_original is a list of arrays. Each array cors to 1 band.
        pixel_info = [time1, my_pid]

#        try:
            bfast_brkpts, ewma_brkpts, ltr_brkpts, polyAlgo_brkpts, winner = \
                process_pixel(tyeardoy, vec_obs_original, pixel_info, tickGap, changes, dist_thresh)
#        except:
#            print 'problem pixel', my_pid
#            continue
    
    return
