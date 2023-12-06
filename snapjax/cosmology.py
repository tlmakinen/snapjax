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
from functools import partial


omega_baryon = jc.Planck15().Omega_b
inference_type = "omegade"


def set_cosmology(omegam, omegade=0.7, w=-1.0):

    omega_baryon = 0.0486
    
    if inference_type == "omegade":
        omegak =  1-omegam-omegade
        w = -1.0

    else:
        omegak = 0.0
        w = w

    cosmo = jc.Planck15(Omega_c=omegam - omega_baryon,
                        Omega_k=omegak,
                        w0=w,
                        h=0.72 # this stays fixed !
                        )
    return cosmo


def integrate_spline_approx(f,right_endpts,npts=100):
    x_min = min(0,*right_endpts)
    x_max = max(right_endpts)
    pad = (x_max-x_min)/10
    x_min-=pad
    x_max+=pad

    x_space = np.linspace(x_min,x_max,npts)

    # Get a spline interpolation of f over x_space
    # with no smoothing (s=0). This is probably the
    # speed bottleneck of this function.
    spl = jc.scipy.interpolate.InterpolatedUnivariateSpline(x_space,f(x_space))

    # the spline can be integrated analytically (fast)
    return np.array([spl.integral(0,right_endpt) for right_endpt in right_endpts])


@jit
def integrand(zba, omegam, omegade, w):
    return 1.0/np.sqrt(
        omegam*(1+zba)**3 + omegade*(1+zba)**(3.+3.*w) + (1.-omegam-omegade)*(1.+zba)**2
    )

# integration of the integrand given above, vmapped over z-axis
@jit
def hubble(z,omegam, omegade,w):
    # method for calculating the integral
    #myfun = lambda z: jc.scipy.integrate.romb(integrand,0., z, args=(omegam,omegade,w)) #[0]
    intg = lambda z: integrand(z, omegam=omegam, omegade=omegade, w=w)
    myfun = lambda z: jc.scipy.integrate.simps(intg, 0., z)
    I = jax.vmap(myfun)(z)
    return I



#@jax.jit
def hubble_single(z, omegam, omegade, w):
  #myfun = lambda zb: jc.scipy.integrate.romb(integrand,0., zb, args=(omegam,omegade,w))
  intg = lambda z: integrand(z, omegam=omegam, omegade=omegade, w=w)
  myfun = lambda z: jc.scipy.integrate.simps(intg, 0., z)
  return myfun(z)


@partial(jax.jit, static_argnums=(6))
def Dlz(omegam, omegade, h, z, w, z_helio, single=False):

    # which inference are we doing ?
    if inference_type == "omegade":
      omegakmag =  np.sqrt(np.abs(1-omegam-omegade))
    else:
      omegakmag = 0.

    if single:
      hubbleint = hubble_single(z, omegam, omegade, w)

    else:
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
@partial(jax.jit, static_argnums=(2))
def muz(cosmo, z, single=False):

    omegam = cosmo.Omega_m
    omegade = cosmo.Omega_de # should have its own attribute
    w = cosmo.w0
    h = cosmo.h # make sure that this is fixed at 0.72
    #h = 0.72
    
    z_helio = z # should this be different ?

    if inference_type == "omegade":
        w = -1.0
    else:
        omegade = 1. - omegam

    return (5.0 * np.log10(Dlz(omegam, omegade, h, z, w, z_helio, single=single))+25.)

def muz_single(omegam, omegade, z, single=True):
    z_helio = z # should this be different ?
    #omegade = 1. - omegam
    w = -1.0 # freeze w
    h = 0.72
    return (5.0 * np.log10(Dlz(omegam, omegade, h, z, w, z_helio, single=single))+25.)