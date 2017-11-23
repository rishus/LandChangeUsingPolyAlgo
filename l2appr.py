#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:10:05 2017

@author: rishu
"""
import numpy as np
import scipy.linalg as la

def bsplvb(t, jhigh, index, x, left_fort):
#  CALCULATES THE VALUE OF ALL POSSIBLY NONZERO B-SPLINES AT  X  OF ORDER
#               JOUT  =  MAX( JHIGH , (J+1)*(INDEX-1) )
#  WITH KNOT SEQUENCE  T .
#
#******  M E T H O D  ******
#  THE RECURRENCE RELATION
#
#                       X - T(I)              T(I+J+1) - X
#     B(I,J+1)(X)  =  -----------B(I,J)(X) + ---------------B(I+1,J)(X)
#                     T(I+J)-T(I)            T(I+J+1)-T(I+1)
#
#  IS USED (REPEATEDLY) TO GENERATE THE (J+1)-VECTOR  B(LEFT-J,J+1)(X),
#  ...,B(LEFT,J+1)(X)  FROM THE J-VECTOR  B(LEFT-J+1,J)(X),...,
#  B(LEFT,J)(X), STORING THE NEW VALUES IN  BIATX  OVER THE OLD.  THE
#  FACTS THAT
#            B(I,1) = 1  IF  T(I) .LE. X .LT. T(I+1)
#  AND THAT
#            B(I,J)(X) = 0  UNLESS  T(I) .LE. X .LT. T(I+J)
#  ARE USED. THE PARTICULAR ORGANIZATION OF THE CALCULATIONS FOLLOWS
#  ALGORITHM  (8)  IN CHAPTER X OF THE TEXT.

#    jout  =  max( jhigh , (j+1)*(index-1) )
#    with open("ltr_python_output.csv", "a") as fh:
#        fh.write('-------------' + '\n')
#        fh.write('x_ll ='+ str(x) + '\n')
#        fh.write('knots: ' + '  '.join([str(i) for i in t]) +'\n ')
#    fh.close()
    biatx = np.asarray([0.0 for i in range(jhigh)])
                                   # Only B_i's that are expected to be non-zero at x
                                   # will be evaluate for the desired x. Other B_i's
                                   # will have zero value anyways, so no need to evaluate.
                                   # B_{left} will be non-zero.
                                   # B_{left-1} MAY be non-zero ... depending on order K
                                   # B_{left+1-K} is the earliest B_i that will be non-zero
                                   # in [t_{left}, t_{left+1})
    deltar = np.asarray([ 0.0 for i in range(20)])
    deltal = np.asarray([ 0.0 for i in range(20)])
    
    # j denotes the current order being evaluated. It will run from j = 1 to K
    if index == 1:
        j = 1
        biatx[0] = 1.0
        if j >= jhigh:  # TODO
            return biatx
        else:
            while (j < jhigh): # TODO
                jp1 = j + 1
                deltar[j-1] = t[left_fort + j -1 ] - x  # left_fort denotes the knot just left of x. This 
                                                       # knot is stored at index left - 1, because 
                                                       # indices in python start from 0.
                deltal[j-1] = x - t[left_fort + 1 -j -1]
#                with open("ltr_python_output.csv", "a") as fh:
#                    fh.write('deltar: ' + '  '.join([str(i) for i in deltar[0:5]]) +'\n ')
#                    fh.write('deltal: ' + '  '.join([str(i) for i in deltal[0:5]]) +'\n ')
#                fh.close()
                saved = 0.0
                for i in range(1,j+1):  # 
#                    with open("ltr_python_output.csv", "a") as fh:
#                        fh.write('i = '+str(i)+'jp1-1 = '+ str(jp1-i)+'\n')
#                        fh.write('deltar(i) =' + str(deltar[i-1]) + ', deltal =' + str(deltal[jp1-i-1]) +'\n ')
#                        fh.write('deltar[i-1]+deltal[jp1-i-1] =' + str((deltar[i-1] + deltal[jp1-i-1])) + '\n')
#                        fh.write('biatx['+str(i-1) + '] =' + str(biatx[i-1]) + '\n')
#                    fh.close()
                    term = float(biatx[i-1]) / (float(deltar[i-1]) + float(deltal[jp1-i-1]))
                    biatx[i-1] = saved + float(deltar[i-1]) * float(term)
                    saved = deltal[jp1-i-1] * term
                    
                biatx[jp1-1] = saved
#                with open("ltr_python_output.csv", "a") as fh:
#                    fh.write( 'x_ll =' + str(x) + ', biatx: ' + '  '.join([str(bx) for bx in biatx]) +'\n ')
#                fh.close()
                j = j+1
    else:  # if index == 2
    # TODO: this part needs to be checked for correctness.
        while (j < jhigh):
            jp1 = j + 1
            deltar[j] = t[left_fort + j -1] - x
            deltal[j] = x - t[left_fort + 1 -j -1]
            saved = 0.0
            for i in range(0,j):
                term = biatx[i] / (deltar[i] + deltal[jp1-i])
                biatx[i] = saved + deltar[i] * term
                saved = deltal[jp1-i] * term
        
            biatx[jp1] = saved
            j += 1        

    return biatx

    
def l2appr(n_dim, K, vec_timestamps, vec_obs, numObs, knots, weight):

#  CONSTRUCTS THE (WEIGHTED DISCRETE) L2-APPROXIMATION BY SPLINES OF ORDER
#  K WITH KNOT SEQUENCE T(1), .., T(N+K) TO GIVEN DATA POINTS
#  (TAU(I),GTAU(I)), I=1,...,NTAU. THE B-SPLINE COEFFICIENTS
#   B C O E F  OF THE APPROXIMATING SPLINE ARE DETERMINED FROM THE
#  NORMAL EQUATIONS USING CHOLESKY'S METHOD.

    # n_dim : dimension of the B-spline space
    # K : order
    # numObs : number of data points
    # number of knots = n_dim + K
    
    bcoeff = np.asarray([ 0.0 for i in range(n_dim)])
    biatx = np.asarray( [ 0.0 for i in range(K)])
    Q = np.zeros((K, n_dim))

    left_py = K-1
    leftmk_py = -1  # left - K
    # Note that left cors. to the index of the knot just left of x, by Dr. Watson's notes indices.
    # The knots t run from t_1 to t_{n_dim+K}. 
    # The python vector t will, however, go from 0 to n_dim+K-1.
    # t_1 is stored at location t[0].
    # t_2 is stored at location t[1].
    # t_K is stored at location t[K-1]
    # t[i] is stored at location t[i-1].
    # If we identify an index 'left' st. t[left] < x < t[left+1], then
    # x is actually in between t_{left} and t_{left+1}
    for ll in range(0, numObs):
#        print 'll =', ll, 'x_ll =', vec_timestamps[ll], 
#        print 'knots[left_py]', knots[left_py], 'vec_timestamps[ll] >= knots[left_py]', vec_timestamps[ll] >= knots[left_py]
        # corner case
        if (left_py == n_dim-1) or (vec_timestamps[ll] < knots[left_py+1]):
            # we want: vec_timestamps(ll) \in (knots(left_fort) , knots(left_fort+1))
            # i.e., vec_timestamps(ll) \in (knots(left_py - 1) , knots(left_py))
            # call bsplvb directly
            left_fort = left_py + 1
            biatx[0:K+1] = bsplvb(knots, K, 1, vec_timestamps[ll], left_fort)
        else:
            # Locate left st. vec_timestamps(ll) \in (knots(left_fort) , knots(left_fort+1))
            # i.e., vec_timestamps(ll) \in (knots(left_py - 1) , knots(left_py))
            while (vec_timestamps[ll] >= knots[left_py+1]):
                left_py += 1
                leftmk_py += 1
            # now call bsplvb
            left_fort = left_py + 1
            biatx[0:K+1] = bsplvb(knots, K, 1, vec_timestamps[ll], left_fort)

        # BIATX(MM-1)   (originally, BIATX(MM) in Fortran)
        # CONTAINS THE VALUE OF B(LEFT-K+MM), MM = 1,2,...,K,  AT TAU(LL).
        # HENCE, WITH DW := BIATX(MM-1)*WEIGHT(LL), (originally, BIATX(MM)*WEIGHT(LL) in Fortran)
        # THE NUMBER DW*GTAU(LL)
        # IS A SUMMAND IN THE INNER PRODUCT
        # (B(LEFT-K+MM),G) WHICH GOES INTO BCOEF(LEFT-K+MM-1)  (originally, BCOEF(LEFT-K+MM) in Fortran)
        # AND THE NUMBER BIATX(JJ)*DW IS A SUMMAND IN THE INNER PRODUCT
        # (B(LEFT-K+JJ),B(LEFT-K+MM)), INTO Q(JJ-MM+1,LEFT-K+MM)
        # SINCE (LEFT-K+JJ) - (LEFT-K+MM) + 1 = JJ - MM + 1 .
        
        # Solving:             min || Ax - b ||
        # So, we solve:       A^tA x = A^t b
        # 
        # Entries of A^t A:
        #                  
        #                  ---
        #                  \
        # [AtA]_{ij} =          B_i(x_l) * B_j(x_j) * w(l)
        #                  /
        #                  ---
        #                  i,j= 
        #
        # AtA is a symmetric banded matrix due to local support of B-splines. 
        # No point storing (or even calculating) all entries.
        # Only nonzero entries are calculated.
        # 
        # Entries of A^t b:
        #    
        for mm in range(1, K+1):
            dw = biatx[mm-1] * weight[ll]
            j = (leftmk_py+1) + mm
            # The rhs < A^t, b >
            bcoeff[j-1] = dw * vec_obs[ll] + bcoeff[j-1]
            i = 1
            for jj in range(mm, K+1):
                Q[i-1,j-1] = biatx[jj-1] * dw +  Q[i-1,j-1]  # j is fixed for currnt x_ll and current mm.
                                                             # only i and jj are incrementing.
                                                             # Q[*, j-1] is getting filled.
                                                             # Then mm increments. j will also increment.
                                                             # Q[*, j_old+1] will get filled.
                                                             #
                                                             # Then next x_ll comes in.
                                                             # Suppose x_ll is in the same knot interval.
                                                             # mm will still run from 1 to K.
                                                             # j, by formula, will again take same values as previous loop.
                                                             # So same columns of Q will get updated.
                                                             # i will also take same values as previous ll-loop.
                                                             # So, same rows of Q will update.
                                                             # Essentially, for fixed [left, left+1), same entries
                                                             # of Q will update.
                                                             # 
                                                             # Suppose x_ll is in a different interval.
                                                             # i.e., left is different, say, left_new = left_old+1
                                                             # Then, j takes 1 higher values than prevoius ll-loops.
                                                             # Meaning, a new (1 up) set of columns is filled.
                                                             # mm and K are always same. So each row is filled again,
                                                             # just in a different column.
                                                             # That's the story of 'matrix filling' here.
                                                             # Figure out the actual working of bandedness etc. -- TODO.
                i += 1


#       Get the CHOLESKY FACTORIZATION FOR banded matrix Q, THEN USE
#       IT TO SOLVE THE NORMAL EQUATIONS
#         C*X = BCOEF
#       FOR X , AND STORE X IN BCOEF
#    print 'In l2appr'
#    print  'n_dim =', n_dim
#    with open("ltr_python_output.csv", "a") as fh:
#        fh.write('writing Q:' + '\n')
#        for i in range(K):
#            fh.write('  '.join([str(Q[i, col]) for col in range(n_dim)] ) + '\n ')
#    fh.close()
    cb = la.cholesky_banded(Q, overwrite_ab=False, lower=True, check_finite=True)
    bcoeff = la.cho_solve_banded((cb, True), bcoeff, overwrite_b=True, check_finite=True)


    return bcoeff    

def bvalue(x, bcoef, jderiv, K, knots, n_dim):
    
    AJ = np.asarray([0.0 for i in range(20)])
    DL = np.asarray([0.0 for i in range(20)])
    DR = np.asarray([0.0 for i in range(20)])
    
    bval = 0.0
    
    if (jderiv >= K):
        
        return bval

#     Original fortran comment:        
#  *** FIND  I   S.T.   1 .LE. I .LT. N+K   AND   T(I) .LT. T(I+1)   AND
#      T(I) .LE. X .LT. T(I+1) . IF NO SUCH I CAN BE FOUND,  X  LIES
#      OUTSIDE THE SUPPORT OF  THE SPLINE  F  AND  BVAL = 0.
#      (THE ASYMMETRY IN THIS CHOICE OF  I  MAKES  F  RIGHTCONTINUOUS)
    
#   But the above approach causes a problem at the right end point of the domain.
#   So, for the right end point, I will look for the interval that satisfies
#       T(I) < x <= T(I+1)
    # Locate left st. vec_timestamps(ll) \in (knots(left_fort) , knots(left_fort+1))
    # i.e., vec_timestamps(ll) \in (knots(left_py - 1) , knots(left_py))
    left_py = K-1
    while ((x >= knots[left_py+1]) or (knots[left_py] >= knots[left_py+1])):
        if (left_py+1 >= n_dim):
            break
        
        left_py += 1
        
            
    if (left_py < 0) or (left_py >= n_dim+K) or (knots[left_py] >= knots[left_py+1]):
        bval = bcoef[left_py]
        return bval

    left_fort = left_py+1
    
#     *** IF K = 1 (AND JDERIV = 0), BVAL = BCOEF(I).
    km1 = K-1
    if km1 <= 0:
        bval = bcoef[left_fort-1]
        return bval

#  *** Store the K B-Spline coefficients relevant for the knot interval
#     (T(left_fort),T(left_fort+1)) in AJ(1),...,AJ(K)
#     and compute DL(J) = X - T(LEFT+1-J),
#     DR(J) = T(LEFT+J) - X, J=1,...,K-1 . 
#     Set any of the AJ not obtainable from input to 0.
#     Set any T.S not obtainable equal to T(1) or
#     to T(N+K) appropriately.

    jcmin_fort = 1
    leftmk_fort = left_fort - K
    if leftmk_fort >= 0:
        for j in range(1,km1+1):
            DL[j-1] = x - knots[left_fort+1-j-1]
    else:
        jcmin_fort = 1 - leftmk_fort
        for j in range(1, left_fort+1):
            DL[j-1] = x - knots[left_fort+1-j-1]
        
        for j in range(left_fort, km1+1):
            AJ[K-j-1] = 0.0
            DL[j-1] = DL[left_fort-1]
    
    jcmax_fort = K
    nmleft_fort = n_dim - left_fort
    if (nmleft_fort >= 0):
        for j in range(1,km1+1):
            DR[j-1] = knots[left_fort + j - 1] - x
    else:
        jcmax_fort = K + nmleft_fort
        for j in range(1,jcmax_fort+1):
            DR[j-1] = knots[left_fort+j-1] - x

        for j in range(jcmax_fort,km1+1):
            AJ[j+1-1] = 0.0
            DR[j-1] = DR[jcmax_fort-1]

    for jc in range(jcmin_fort, jcmax_fort+1):
        AJ[jc-1] = bcoef[leftmk_fort + jc -1]
        
#      *** DIFFERENCE THE COEFFICIENTS  JDERIV  TIMES ********
    if (jderiv != 0):
        for j in range(1, jderiv+1):
            kmj = K-j
            fkmj = float(kmj)
            ilo_fort = kmj
            for jj in range(1,kmj+1):
                AJ[jj-1] = ((AJ[jj] - AJ[jj-1])/(DL(ilo_fort-1)+DR(jj-1)))*fkmj
                ilo_fort = ilo_fort-1
    
#  *** COMPUTE VALUE AT  X  IN (T(I),T(I+1)) OF JDERIV-TH DERIVATIVE,
#     GIVEN ITS RELEVANT B-SPLINE COEFFS IN AJ(1),...,AJ(K-JDERIV).
    if (jderiv == km1):
        bval = AJ[0]
        return
    jtemp1_fort = jderiv + 1
    for j in range(jtemp1_fort, km1+1):
        kmj = K - j
        ilo_fort = kmj
        for jj in range(1, kmj+1):
            AJ[jj-1] = (AJ[jj+1-1]*DL[ilo_fort-1] + AJ[jj-1]*DR[jj-1])/(DL[ilo_fort-1]+DR[jj-1])
            ilo_fort -= ilo_fort - 1
    
    bval = AJ[0]
    
    return bval    
