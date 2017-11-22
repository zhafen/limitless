########################################################################                                                                                                     
#
# Name: survival_analysis.py
# Author: Zachary Hafen (zachary.h.hafen@gmail.com)
# Purpose: Tools for performing survival analysis
#                                                                                                                                                                           
######################################################################## 

import copy
import math
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.stats
import scipy.special 
import scipy.signal
from scipy.io.idl import readsav
import sys

########################################################################

def calc_number_at_risk(d, r):
  'Calculate the number of data points at risk'
  
  Y = np.zeros(d.shape)
  for i, r_i in enumerate(r):

    Y[i] = d[i:].sum() + r[i:].sum()
    
  return Y

########################################################################

def inverse_calc_number_at_risk(d, c):
  '''Calculate the number of data points that were at risk.
  Opposite of Y.'''
  
  W = np.zeros(d.shape)
  for i, c_i in enumerate(c):

    W[i] = d[:i+1].sum() + c[:i+1].sum()
    
  return W

########################################################################

def kaplan_meier_estimator(time, t, d, Y):
  'From Kaplan&Meier1958'
  
  # Calculate the fraction
  frac_remaining = 1. - d/Y
  
  # No deaths yet
  if time <= t[0]:
    return 1.
  elif time >= t[-1]:
  # If the time value is greater than all the times, return the total product
    return frac_remaining.prod()
  
  # The index of the first time that's greater than the specified time, minus one
  time_index = next(i for i, value in enumerate(t) if value > time) - 1
  
  return frac_remaining[:time_index].prod()

########################################################################

def inverse_kaplan_meier_estimator(time, t, d, W):
  'Like the Kaplan-Meier estimator, but uses left censored data and calculates the CDF instead.'

  # Calculate the fraction
  frac_remaining = 1. - d/W
  
  # If at the top
  if time >= t[-1]:
    return 1.
  elif time <= t[0]:
  # If the time value is less than all the times, return the total product
    return 0.
  
  # The index of the first time that's greater than the specified time, minus one
  time_index = next(i for i, value in enumerate(t) if value > time) - 1
  
  return frac_remaining[time_index:].prod()

########################################################################

def calc_p_ij(i, j, S_k):
  'Get the probability of being between t_j and t_j-1 given T<t_i. (Requires j<= i)'
  
  # Case for when dealing with c_0
  if i == 0:
    return 1.
  # DEBUG
  elif i == len(S_k):
    bottom_term = 1 - S_k[i - 1]
  else:
  # Probability of being <t_i
    bottom_term = 1 - S_k[i]
  
  # Case for when there's no j -1
  if j == 0:
    top_term = 1 - S_k[j]
  # DEBUG
  elif j == len(S_k):
    top_term = S_k[j - 2] - S_k[j - 1]
  # Probability of being between t_j and t_j-1
  else:
    top_term = S_k[j - 1] - S_k[j]
    
  # DEBUG
  if bottom_term == 0:
    bottom_term = 0.0001
  
  p_ij = top_term/bottom_term
  
#   if math.isnan(p_ij) or (p_ij == np.inf):
    # DEBUG
#     print ''
#     print 'S_k =', S_k
#     print 'i, j = {}, {}'.format(i, j)
#     print 'top_term =', top_term
#     print 'bottom_term =', bottom_term
  
  return p_ij

########################################################################

def update_d(d, c, S_k):
  'Update the estimate for the detections using a given survival function.'
  
  d_new = []
  for j in range(len(d)+1):
    
    # Skip over the first loop, since d is defined to start at index one
    if j == 0:
      continue
    
    # Get the probabilities that the censored data will fall in different bins
    p_ij = np.array([calc_p_ij(i, j, S_k) for i in range(len(d)+1)])
    
    # DEBUG
#     print ''
#     print 'j = {}'.format(j)
#     print p_ij
    
    # Estimate the contribution from left censored data
    # Be careful about the indices. Note the j-1. That's because d, c, etc all start at index one
    censored_data_contr = (p_ij[1:]*c)[j-1:].sum()

    # Get the new estimate
    d_j_add = d[j-1] + censored_data_contr
    d_new.append(d_j_add)
    
  d_new = np.array(d_new)
  
  return d_new

########################################################################

class survival_function_estimator(object):
  'Estimate a survival function using the Turnbull(1974) algorithm.'
  
  def __init__(self, t, c, d, r, keep_S_ks=False, keep_other_ks=False, convergence_condition=1e-6,\
              max_iterations=1e3, S_0='KaplanMeier'):

    self.t = t
    
    # ICs
    d_k = copy.deepcopy(d)
    convergence_condition_satisfied = False

    if keep_S_ks:
      self.S_ks = []
      
    if keep_other_ks:
      self.d_ks = []
      self.Y_ks = []

    for k in range(int(max_iterations)):
      
      # Get the number at risk
      Y_k = calc_number_at_risk(d_k, r)

      # Calculate initial estimate for S_k
      if k == 0:
        if (S_0 == 'KaplanMeier'):
          S_k = np.array([kaplan_meier_estimator(t_, t, d_k, Y_k) for t_ in self.t])
        else:
          S_k = S_0
      # All other S_k
      else:
        S_k = np.array([kaplan_meier_estimator(t_, t, d_k, Y_k) for t_ in self.t])

      # Evaluate if we're finished
      if k == 0:
        evaluate_if_finished = False
      elif k == 1 and S_0 != 'KaplanMeier':
        evaluate_if_finished = False
      else:
        evaluate_if_finished = True
      if evaluate_if_finished:
        if keep_other_ks:
          self.d_ks.append(d_k)
          self.Y_ks.append(Y_k)

        # Calculate a measure of how far this new version is from the previous one
        sum_of_square_diffs = ((S_k - S_k_prev)**2.).sum()

        # Return if condition reached
        if sum_of_square_diffs <= convergence_condition:
          convergence_condition_satisfied = True
          break

      # Update d
      d_k = update_d(copy.deepcopy(d), c, S_k)

      # Save the previous version of S_k
      S_k_prev = S_k

      if keep_S_ks:
        self.S_ks.append(S_k)

    # Raise an error if not converged
    if not convergence_condition_satisfied:
      exception_string1 = 'Failed to converge after {} iterations \n'.format(k)
      exception_string2 = \
      'Did not reach convergence condition of sum_of_square_diffs={} \n'.format(convergence_condition)
      exception_string3 = 'sum_of_square_diffs={}'.format(sum_of_square_diffs)
      exception_string = exception_string1 + exception_string2 + exception_string3
      raise Exception(exception_string)

    # Print information
    print 'Took {} iterations to converge.'.format(k)
    print 'Final sum_of_square_diffs from previous iteration = {:.3g}'.format(sum_of_square_diffs)

    self.S_k = S_k
    self.d_k = d_k
    self.Y_k = Y_k

  ########################################################################
    
  def survival_fn_plot_array(self, t_plot):
    
    return np.array([kaplan_meier_estimator(t_, self.t, self.d_k, self.Y_k) for t_ in t_plot])

  ########################################################################
  
  def smoothed_survival_fn(self, t_plot, window_size=101, polynomial_order=1):
    
    S_plot = self.survival_fn_plot_array(t_plot)

    # Smooth further using a savgol filter
    S_savgol = sp.signal.savgol_filter(S_plot, window_size, polynomial_order)

    return S_savgol

  ########################################################################

  def estimate_pdf(self, t_plot, **kw_args):

    S_smoothed = self.smoothed_survival_fn(t_plot, **kw_args)

    return -np.gradient(S_smoothed, t_plot[1] - t_plot[0])
