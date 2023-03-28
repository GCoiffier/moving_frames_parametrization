from enum import Enum
from dataclasses import dataclass
import numpy as np

import mouette as M
from .instance import Instance

##### Distortion #####

class Distortion(Enum):
    """Description of the different energies used a a distortion in the optimization.
    """
    NONE = 0
    LSCM = 1
    SHEAR = 2
    ARAP = 3
    SCALE = 4

    @staticmethod
    def from_string(s :str):
        if "lscm" in s.lower():
            return Distortion.LSCM
        if "shear" in s.lower():
            return Distortion.SHEAR
        if "arap" in s.lower():
            return Distortion.ARAP
        if "scale" in s.lower():
            return Distortion.SCALE
        return Distortion.NONE

##### Init Mode #####

class InitMode(Enum):
    """
    Initialization mode
    """
    ZERO = 0
    SMOOTH = 1
    CURVATURE = 2
    RANDOM = 3

    @staticmethod
    def from_string(s :str):
        if "zero" in s.lower():
            return InitMode.ZERO
        if "smooth" in s.lower():
            return InitMode.SMOOTH
        if "curv" in s.lower():
            return InitMode.CURVATURE
        if "random" in s.lower():
            return InitMode.RANDOM
        raise Exception(f"InitMode {s} not recognized")

##### Default running options #####

@dataclass
class VerboseOptions:
    output_dir : str = ""
    logger_verbose : bool = True
    qp_solver_verbose : bool = False
    optim_verbose : bool = True
    snapshot_freq : int = 0
    log_freq : int = 0
    tqdm : bool = True

@dataclass
class Options:
    distortion : Distortion = Distortion.LSCM
    features : bool = False
    initMode : InitMode = InitMode.ZERO
    optimFixedFF : bool = False
    n_iter_max : int = 1000
    dist_schedule : list = None

    def set_schedule(self,sch : list = None):
        if sch is None:
            if self.distortion == Distortion.NONE:
                self.dist_schedule = []
            elif self.distortion == Distortion.LSCM:
                self.dist_schedule = [x for x in np.logspace(2, -2, 5)]
            elif self.distortion in [Distortion.ARAP, Distortion.SHEAR]:
                self.dist_schedule = [x for x in np.logspace(2, -4, 7)]
            elif self.distortion == Distortion.SCALE:
                self.dist_schedule = [x for x in np.logspace(2, -4, 7)]
        else:
            self.dist_schedule = sch

class Worker(M.Logger):
    """An abstract worker applying functions on a parametrization problem instance"""

    def __init__(self, 
        name : str, 
        instance : Instance, 
        options : Options = Options(),
        verbose_options : VerboseOptions = VerboseOptions()):
        
        self.instance = instance
        self.options = options
        self.verbose_options = verbose_options
        M.Logger.__init__(self, name, self.verbose_options.logger_verbose)