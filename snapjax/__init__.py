__version__ = "0.1.0"

from functools import partial
from .cosmology import *
#from .sampler import *
from .cosmology_container import SNiaCosmology
import snapjax.selection_effects as selection

SNIa_fiducial = partial(SNiaCosmology, 
                        Omega_m=0.3,
                        Omega_de=0.7,
                        h=0.72,
                        n_s=0.9667,
                        sigma8=0.8159,
                        w0=-1.0,
                        wa=0.0,
                    )


def set_cosmology(omegam, omegade=0.7, w=-1.0):
    """_summary_

    Args:
        omegam (float): Matter density fraction
        omegade (float, optional): Dark energy density fraction. Defaults to 0.7.
        w (float, optional): Dark energy equation of state. Defaults to -1.0.

    Returns:
        snapjax.SNiaCosmology s: cosmology container object (pytree)
    """

    if inference_type == "omegade":
        omegak =  1-omegam-omegade
        w = -1.0

    else:
        omegak = 0.0
        w = w

    cosmo = SNIa_fiducial(Omega_m=omegam,
                       Omega_de=omegade,
                       w0=w,
                       h=0.72 # this stays fixed !
                       )
    return cosmo
