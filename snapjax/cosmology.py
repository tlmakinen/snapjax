import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import torch
import jax
from jax import lax
import jax_cosmo as jc
import scipy.constants as cnst

import matplotlib.pyplot as plt


# the integrand in the calculation of mu from z,cosmology
@jit
def integrand(zba, omegam, omegade, w):
    return 1.0/np.sqrt(
        omegam*(1+zba)**3 + omegade*(1+zba)**(3.+3.*w) + (1.-omegam-omegade)*(1.+zba)**2
    )

# integration of the integrand given above, vmapped over z-axis
@jit
def hubble(z,omegam, omegade,w):
    # method for calculating the integral
    myfun = lambda z: jc.scipy.integrate.romb(integrand,0., z, args=(omegam,omegade,w)) #[0]
    I = jax.vmap(myfun)(z)
    return I


@jit
def Dlz(omegam, omegade, h, z, w, z_helio):

    # which inference are we doing ?
    if inference_type == "omegade":
      omegakmag =  np.sqrt(np.abs(1-omegam-omegade))  
    else:
      omegakmag = 0.

    hubbleint = hubble(z,omegam,omegade,w)
    condition1 = (omegam + omegade == 1) # return True if = 1 
    condition2 = (omegam + omegade > 1.)

    #if (omegam+omegade)>1:
    def ifbigger(omegakmag):
      return (cnst.c*1e-5 *(1+z_helio)/(h*omegakmag)) * np.sin(hubbleint*omegakmag)

    # if (omegam+omegade)<1:
    def ifsmaller(omegakmag):
      return cnst.c*1e-5 *(1+z_helio)/(h*omegakmag) *np.sinh(hubbleint*omegakmag)   

    # if (omegam+omegade==1):
    def equalfun(omegakmag):
      return cnst.c*1e-5 *(1+z_helio)* hubbleint/h

    # if not equal, default to >1 condition
    def notequalfun(omegakmag):
      return lax.cond(condition2, true_fun=ifbigger, false_fun=ifsmaller, operand=omegakmag)

    distance = lax.cond(condition1, true_fun=equalfun, false_fun=notequalfun, operand=omegakmag)

    return distance


# muz: distance modulus as function of params, redshift
@jit
def muz(omegam, w, z):
    z_helio = z # should this be different ?
    omegade = 1. - omegam
    #w = -1.0 # freeze w
    h = 0.72
    return (5.0 * np.log10(Dlz(omegam, omegade, h, z, w, z_helio))+25.)