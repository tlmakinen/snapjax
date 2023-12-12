import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import jax
from jax import lax
import jax_cosmo as jc
import scipy.constants as cnst


import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC

import matplotlib.pyplot as plt

from snapjax.cosmology import *

@jit
def log_uniform(r, log_min_val, log_max_val):
    """log-uniform prior
      given point sampled from unif(0, 1)
      return point sampled on a uniform log scale from min_val to max_val
      """
    point = 10.0 ** (log_min_val + r * (log_max_val - log_min_val))

    return point


## define bahamas HMC model
def bahamas(z, data, Cov, snID, ndat):

    # globals
    OmM = numpyro.sample('OmM', dist.Uniform(low=0.0, high=1.0))
    w = numpyro.sample('w', dist.Uniform(low=-2.0, high=0.0))
    #OmDE = numpyro.sample('OmDE', dist.Uniform(low=0.0, high=2.0)) # for OmDE inference

    α = numpyro.sample('α', dist.Uniform(0.0, 1.0))
    β = numpyro.sample('β', dist.Uniform(0.0, 4.0))

    R_B = numpyro.sample('R_B', dist.Uniform(0.0, 5.0)) # should equal β in toy example


    # mB hypers
    σ_res =  numpyro.sample('σ_res', dist.TransformedDistribution(
                      dist.Uniform(-3.0, 0.0), 
                      dist.transforms.ExpTransform()))

    M0 = numpyro.sample('M0', dist.Normal(-19.3, 2.0))

    # stretch hypers
    xstar = numpyro.sample('xstar', dist.Normal(loc=0.0,scale=0.1))
    Rx =  numpyro.sample('Rx', dist.TransformedDistribution(
                      dist.Uniform(-5.0, 2.0), 
                      dist.transforms.ExpTransform()))


    # color hypers
    cstar = numpyro.sample('cstar', dist.Normal(loc=0.0,scale=0.1))

    Rc = numpyro.sample('Rc', dist.TransformedDistribution(
                      dist.Uniform(-5.0, 2.0), 
                      dist.transforms.ExpTransform()))
    
    # SIMPLE BAYESN PARAMETERS
    τ = numpyro.sample("τ", dist.Uniform(0.0, 10.0)) # for dust

    unique_sn_IDs = snID
    n_sne = len(unique_sn_IDs)

    # now latents

    with numpyro.plate("plate_i", n_sne) as idx:
        c_ = numpyro.sample('c', dist.Normal(cstar, Rc))
        x1_ = numpyro.sample('x1', dist.Normal(xstar, Rx))
        M_ = numpyro.sample('M', dist.Normal(M0, σ_res))
        E_ = numpyro.sample('E_', dist.Exponential(rate=1./τ))



   # myM = -(muz(true_omegam, true_w, z) - true_alpha*datadf["x1"] + true_beta*datadf["c"]) + datadf['mb']
   # datadf['mb'] = + muz + alpha*x1 - beta*c + M

    # compute mB [IMPORTANT BIT]
    mB_ = muz(OmM, w, z[snID]) - α*x1_[snID] + β*c_[snID] + M_[snID] + (R_B - β)*E_[snID]


    # assemble means for obs variable draw
    _obsloc = np.stack((c_,x1_,mB_), axis=1)

    # now sample observed values with measured covariance
    with numpyro.plate("plate_i", n_sne): #as idx:
        numpyro.sample('obs', dist.MultivariateNormal(_obsloc, Cov), obs=data)
