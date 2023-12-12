__version__ = "0.1.0"

from functools import partial
from .cosmology import *
from .sampler import *
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
