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
  
    zH = z.copy()

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
In order to run Numpyro's GPU-enabled HMC sampler, we need access to gradients of all 
