# snapjax
a fast, flexible SuperNova Analysis Pipeline for Cosmology in Jax

Type Ia Supernovae (SNIa) are standard candles used to infer the expansion rate of the universe by measuring distances to large redshifts. We can write down a Bayesian Hierarchical Model (BHM, based on [BAHAMAS](https://arxiv.org/pdf/1510.05954.pdf)) which incorporates cosmology and SNIa properties to estimate cosmological parameters from a set of observed SNIa. 

If we code the cosmological calculations *differentiably*, we can perform HMC sampling efficienty in `Jax` and `Numpyro`, and make the code easy to read and modular to boot. This is what `snapjax` is all about.


# run in-browser
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UoxzW9FWUZlQ6mfxX27CELKAH5t5irZB?usp=sharing)

Run the whole (toy) analysis in Google Colab !

<!-- # requirements -->


<!--  # setup -->


# a quick example
`snapjax` is designed to be modular within the `Numpyro` framework. For example, we can code the entire vanilla [BAHAMAS](https://arxiv.org/pdf/1510.05954.pdf) model in just one block of code, say for *w*-Ω inference

```python
def bahamas(z, data, Cov, snID, ndat):

    # globals
    OmM = numpyro.sample('OmM', dist.Uniform(low=0.0, high=1.0))
    w = numpyro.sample('w', dist.Uniform(low=-2.0, high=0.0))

    α = numpyro.sample('α', dist.Uniform(0.0, 1.0))
    β = numpyro.sample('β', dist.Uniform(0.0, 4.0))

    # mB hypers
    σ_res_sq = numpyro.sample('σ_res_sq', dist.InverseGamma(0.003, 0.003))
    M0 = numpyro.sample('M0', dist.Normal(-19.3, 2.0))
    
    # stretch hypers
    xstar = numpyro.sample('xstar', dist.Normal(loc=0.0,scale=0.1))
    _rx = numpyro.sample('_rx', dist.Uniform())
    Rx =  log_uniform(_rx, -5.0, 2.0)
    
    # color hypers
    cstar = numpyro.sample('cstar', dist.Normal(loc=0.0,scale=0.1))
    _rc = numpyro.sample('Rc', dist.Uniform())
    Rc = log_uniform(_rc, -5.0, 2.0)

    unique_sn_IDs = snID
    n_sne = len(unique_sn_IDs)

    # now latents
    with numpyro.plate("plate_i", n_sne) as idx:
        c_ = numpyro.sample('c', dist.Normal(cstar, Rc))
        x1_ = numpyro.sample('x1', dist.Normal(xstar, Rx))
        M_ = numpyro.sample('M', dist.Normal(M0, np.sqrt(σ_res_sq)))

    # compute mB [IMPORTANT BIT]
    mB_ = muz(OmM, w, z[snID]) - α*x1_[snID] + β*c_[snID] + M_[snID]


    # assemble means for obs variable draw
    _obsloc = np.stack((c_,x1_,mB_), axis=1)

    # now sample observed values with measured covariance
    with numpyro.plate("plate_i", n_sne): #as idx:
        numpyro.sample('obs', dist.MultivariateNormal(_obsloc, Cov), obs=data)
```


# how it works
In order to run Numpyro's GPU-enabled HMC sampler, we need access to gradients of all cosmological calculations made for calculating the distance modulus, μ(z). We can exploit `Jax`'s handy autodiff features employed in the [jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo) library. Here are the functions:

```python
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import jax
from jax import lax
import jax_cosmo as jc
import scipy.constants as cnst
```

We start by defining the integrand, making use of the `jax-cosmo.scipy` library

```python

inference_type = 'w'

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
    
```

Now we define our conditional functions to define our `sinn` function, which depends on the curvature parameter, `omegakmag`. We have to define these functions in this way to allow autodifferentiation through `Jax`'s LAX backend:

```python
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
```
