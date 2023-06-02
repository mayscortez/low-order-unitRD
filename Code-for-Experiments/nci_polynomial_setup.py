from dataclasses import replace
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from math import log, ceil
import pandas as pd
import seaborn as sns
import nci_linear_setup as ncls
from scipy import interpolate, special

# Scale down the effects of higher order terms
a1 = 1      # for linear effects
a2 = 1    # for quadratic effects
a3 = 1   # for cubic effects
a4 = 1   # for quartic effects

# Define f(z)
f_linear = lambda alpha, z, gz: alpha + a1*z
f_quadratic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz)
f_cubic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3)
f_quartic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3) + a4*np.power(gz,4)

def ppom(beta, C, alpha):
  '''
  Returns k-degree polynomial potential outcomes function fy
  
  f (function): must be of the form f(z) = alpha + z + a2*z^2 + a3*z^3 + ... + ak*z^k
  C (np.array): weighted adjacency matrix
  alpha (np.array): vector of null effects
  '''
  # n = C.shape[0]
  # assert np.all(f(alpha, np.zeros(n), np.zeros(n)) == alpha), 'f(0) should equal alpha'
  #assert np.all(np.around(f(alpha, np.ones(n)) - alpha - np.ones(n), 10) >= 0), 'f must include linear component'

  if beta == 0:
      return lambda z: alpha + a1*z
  elif beta == 1:
      f = f_linear
      return lambda z: alpha + a1*C.dot(z)
  else:
      g = lambda z : C.dot(z) / np.array(np.sum(C,1)).flatten()
      if beta == 2:
          f = f_quadratic
      elif beta == 3:
          f = f_cubic
      elif beta == 4:
          f = f_quartic
      else:
          print("ERROR: invalid degree")
      return lambda z: f(alpha, C.dot(z), g(z)) 

SNIPE_beta = lambda n,y,w : np.sum(y*w)/n 

def SNIPE_weights(n, p, A, z, beta):
  '''
  Compute the weights w_i(z) for each population unit

  n (int): population size
  p (float): treatment probability
  A (scipy csr array): adjacency matrix in scipy csr format
  z (numpy array): treatment vector
  beta (int): degree of the potential outcomes model

  Returns a numpy array W where the i-th entry is the weight w_i(z) associated to unit i
  '''
  treated_neighb = A.dot(z)
  control_neighb = A.dot(1-z)
  W = np.zeros(n)
  for i in range(n):
    w = 0
    a_lim = min(beta,int(treated_neighb[i]))
    for a in range(a_lim+1):
      b_lim = min(beta - a,int(control_neighb[i]))
      for b in range(b_lim+1):
        w = w + ((1-p)**(a+b) - (-p)**(a+b)) * p**(-a) * (p-1)**(-b) * special.binom(treated_neighb[i],a)  * special.binom(control_neighb[i],b)
    W[i] = w

  return W

def poly_interp_splines(n, P, sums, spltyp = 'quadratic'):
  '''
  Returns estimate of TTE using spline polynomial interpolation 
  via scipy.interpolate.interp1d

  n (int): popluation size
  P (numpy array): sequence of probabilities p_t
  sums (numpy array): sums of outcomes at each time step
  spltyp (str): type of spline, can be 'quadratic, or 'cubic'
  '''
  assert spltyp in ['quadratic', 'cubic'], "spltyp must be 'quadratic', or 'cubic'"
  f_spl = interpolate.interp1d(P, sums, kind=spltyp, fill_value='extrapolate')
  TTE_hat = (1/n)*(f_spl(1) - f_spl(0))
  return TTE_hat

def poly_interp_linear(n, P, sums):
  '''
  Returns two estimates of TTE using linear polynomial interpolation 
  via scipy.interpolate.interp1d
  - the first is with kind = 'linear' (as in... ?)
  - the second is with kind = 'slinear' (as in linear spline)

  n (int): popluation size
  P (numpy array): sequence of probabilities p_t
  sums (numpy array): sums of outcomes at each time step
  '''

  #f_lin = interpolate.interp1d(P, sums, fill_value='extrapolate')
  f_spl = interpolate.interp1d(P, sums, kind='slinear', fill_value='extrapolate')
  #TTE_hat1 = (1/n)*(f_lin(1) - f_lin(0))
  TTE_hat2 = (1/n)*(f_spl(1) - f_spl(0))
  #return TTE_hat1, TTE_hat2
  return TTE_hat2


def poly_regression_prop(beta, y, A, z):
  '''
  Returns an estimate of the TTE using polynomial regression using
  numpy.linalg.lstsq

  beta (int): degree of polynomial
  y (numpy array): observed outcomes
  A (square numpy array): network adjacency matrix
  z (numpy array): treatment vector
  '''
  n = A.shape[0]

  if beta == 0:
      X = np.ones((n,2))
      X[:,1] = z
  else:
      X = np.ones((n,2*beta+1))
      count = 1
      treated_neighb = (A.dot(z)-z)/(np.array(A.sum(axis=1)).flatten()-1+1e-10)
      for i in range(beta):
          X[:,count] = np.multiply(z,np.power(treated_neighb,i))
          X[:,count+1] = np.power(treated_neighb,i+1)
          count += 2

  v = np.linalg.lstsq(X,y,rcond=None)[0]
  return np.sum(v)-v[0]

def poly_regression_prop_cy(beta, y, A, z):
  n = A.shape[0]
  X = np.ones((n,2*beta+2))
  z = z.reshape((n,1))
  treated_neighb = (A.dot(z)-z)/(np.array(A.sum(axis=1)).flatten()-1+1e-10)
  # temp = 1
  # for i in range(beta+1):
  #     X[:,i] = np.multiply(z,temp)
  #     X[:,beta+1+i] = np.multiply(1-z,temp)
  #     temp = temp * treated_neighb
  treated_neighb = np.power(treated_neighb.reshape((n,1)), np.arange(beta+1).reshape((1,beta+1)))
  X[:,:beta+1] = z.dot(treated_neighb)
  X[:,beta+1:] = (1-z).dot(treated_neighb)

  v = np.linalg.lstsq(X,y,rcond=None)[0]
  return np.sum(v[:beta+1])-v[beta+1]

def poly_regression_num(beta, y, A, z):
  '''
  Returns an estimate of the TTE using polynomial regression using
  numpy.linalg.lstsq

  beta (int): degree of polynomial
  y (numpy array): observed outcomes
  A (square numpy array): network adjacency matrix
  z (numpy array): treatment vector
  '''
  n = A.shape[0]

  if beta == 0:
      X = np.ones((n,2))
      X[:,1] = z
  else:
      X = np.ones((n,2*beta+1))
      count = 1
      treated_neighb = (A.dot(z)-z)
      for i in range(beta):
          X[:,count] = np.multiply(z,np.power(treated_neighb,i))
          X[:,count+1] = np.power(treated_neighb,i+1)
          count += 2

  # least squares regression
  v = np.linalg.lstsq(X,y,rcond=None)[0]

  # Estimate TTE
  count = 1
  treated_neighb = np.array(A.sum(axis=1)).flatten()-1
  for i in range(beta):
      X[:,count] = np.power(treated_neighb,i)
      X[:,count+1] = np.power(treated_neighb,i+1)
      count += 2
  TTE_hat = np.sum((X @ v) - v[0])/n
  return TTE_hat

def poly_regression_num_cy(beta, y, A, z):
  n = A.shape[0]

  X = np.ones((n,2*beta+2))
  z = z.reshape((n,1))
  treated_neighb = (A.dot(z)-z)
  # temp = 1
  # for i in range(beta+1):
  #     X[:,i] = np.multiply(z,temp)
  #     X[:,beta+1+i] = np.multiply(1-z,temp)
  #     temp = temp * treated_neighb
  treated_neighb = np.power(treated_neighb.reshape((n,1)), np.arange(beta+1).reshape((1,beta+1)))
  X[:,:beta+1] = z.dot(treated_neighb)
  X[:,beta+1:] = (1-z).dot(treated_neighb)

  # least squares regression
  v = np.linalg.lstsq(X,y,rcond=None)[0]

  # Estimate TTE
  X = np.zeros((n,2*beta+2))
  deg = np.array(A.sum(axis=1)).flatten()-1
  # temp = 1
  # for i in range(beta+1):
  #     X[:,i] = temp
  #     temp = temp * deg
  X[:,:beta+1] = np.power(deg.reshape((n,1)), np.arange(beta+1).reshape((1,beta+1)))
  TTE_hat = np.sum((X @ v) - v[beta+1])/n

  
  return TTE_hat

def SNIPE_beta_old(n, p, y, A, z, beta):
  # n = z.size
  # z = z.reshape((n,1))
  treated_neighb = A.dot(z)
  control_neighb = A.dot(1-z)
  est = 0
  for i in range(n):
    w = 0
    a_lim = min(beta,int(treated_neighb[i]))
    for a in range(a_lim+1):
      b_lim = min(beta - a,int(control_neighb[i]))
      for b in range(b_lim+1):
        w = w + ((1-p)**(a+b) - (-p)**(a+b)) * p**(-a) * (p-1)**(-b) * special.binom(treated_neighb[i],a)  * special.binom(control_neighb[i],b)
    est = est + y[i]*w

  return est/n
