import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
#import torch
import jax
from jax import lax
import jax_cosmo as jc
import scipy.constants as cnst


#eps,gmB,gc,gx1 = (29.96879778, -1.34334963,  0.45895811,  0.06703621)

#selection_param=(gc, gx1, gmB, eps)

#gc,gx1,gmB,eps = (-1.7380184749999987, 0.07955165160000005, -2.60483735, 62.15588654)

#@jax.jit
def log_indiv_selection_fn(phi_i, selection_param=np.array([gc, gx1, gmB, eps])):
    coefs = np.array(selection_param)
    position = np.concatenate([phi_i, np.ones(1)]) #np.array([*phi_i, 1])
    argument = np.dot(coefs, position)
    return jax.scipy.stats.norm.logcdf(np.sqrt(np.pi/8)*argument) # must be a logcdf so it dies/grows to 0/1 at the right speed

#@jax.jit
def log_latent_marginalized_indiv_selection_fn(mu_i, param,
                                               selection_param=(gc, gx1, gmB, eps)):
    alpha, beta = param[0:2]
    rx, rc, sigma_res = param[2:5]
    cstar, xstar, mstar = param[5:8]
    gc, gx, gm, eps = selection_param


    coefs = np.array([gc + gm * beta, gx - gm * alpha, gm, eps])
    denominator = np.sqrt((8 / np.pi)
                          + (coefs[0] * rc)**2
                          + (coefs[1] * rx)**2
                          + (coefs[2] * sigma_res)**2)

    new_coefs = coefs / denominator
    position = np.array([cstar, xstar, mu_i + mstar, 1])
    argument = np.dot(new_coefs, position)

    return jax.scipy.stats.norm.logcdf(argument)

@jax.jit
def supernova_redshift_pdf(z, b=1.5, z_min=0.0, z_max=3.0):
  # if rate is \propto (1 + z)^b => pdf(z, b) = b * (1 + z)^(b-1)
  x = (1 + z) # switch to (1+z) coords
  zmin = 1 + z_min
  zmax = 1 + z_max
  scale = (zmax - zmin)
  x = (x - zmin) / scale # minmax scaling

  # redshift dist follows (1 + z)**1.5
  # => need pdf to follow f(x, alpha) = a x^(a-1); 0 < x < 1
  return b * x**(b - 1.0) / scale


# for redshift trapezoidal integration
z_array = np.linspace(0.001, 3.0, num=120)

def redshift_marginalization_integrand(z, param,
                                       cosmo,
                                       selection_param,
                                       ):

    
    # do trapz integration
    mu = (muz(cosmo, z, single=True)) # TODO: Should the z's be the same?
    myfunc = lambda m: log_latent_marginalized_indiv_selection_fn(m, param=param, selection_param=selection_param)
    result = myfunc(mu)
    return np.exp(result) * (supernova_redshift_pdf(z))

#@jax.jit
def log_redshift_marginalized_indiv_selection_fn(param, cosmo, selection_param):
    myfunc = lambda z: redshift_marginalization_integrand(z, param, cosmo, selection_param)
    curve_to_integrate = jax.vmap(myfunc)(z_array)

    return np.log(jax.scipy.integrate.trapezoid(y=curve_to_integrate, x=z_array)) #z_array[:, np.newaxis], 4, axis=-1))
    #return (jc.scipy.integrate.romb(redshift_marginalization_integrand, 0, 3.0, args=(param, cosmo_param, selection_param)))

def rubin_log_correction(param, selection_param, phi, mu):
    log_numerator = [log_indiv_selection_fn(phi_i, selection_param) for phi_i in phi]
    log_denominator= [log_latent_marginalized_indiv_selection_fn(mu_i, param, selection_param) for mu_i in mu]
    return np.sum(log_numerator) - np.sum(log_denominator)

@jax.jit
def vincent_log_correction(param, cosmo, selection_param, phi, ndat):
    indv_fn = lambda phi: log_indiv_selection_fn(phi, selection_param=selection_param)
    log_numerator = jax.vmap(indv_fn)(phi) #[log_indiv_selection_fn(phi_i, selection_param) for phi_i in phi]

    return (np.sum(log_numerator) - ndat * log_redshift_marginalized_indiv_selection_fn(param, cosmo, selection_param))