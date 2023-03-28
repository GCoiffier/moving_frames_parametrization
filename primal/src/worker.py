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
    ID = 2
    SHEAR = 3
    ISO = 4
    NORMAL = 5

    CONF_SCALE = 6
    
    LSCM_B = 7
    ID_B = 8
    ID_C = 9

    RIGID = 10

    @staticmethod
    def from_string(s:str):
        if "lscmb" in s.lower():
            return Distortion.LSCM_B

        if "lscm" in s.lower():
            return Distortion.LSCM
        
        if "rigid" in s.lower():
            return Distortion.RIGID

        if "idb" in s.lower():
            return Distortion.ID_B
        
        if "idc" in s.lower():
            return Distortion.ID_C

        if "id" in s.lower():
            return Distortion.ID

        if "shear" in s.lower():
            return Distortion.SHEAR

        if "iso" in s.lower():
            return Distortion.ISO
        
        if "conf" in s.lower():
            return Distortion.CONF_SCALE

        if "normal" in s.lower():
            return Distortion.NORMAL
    
        return Distortion.NONE


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
    initFixedFF: bool = False
    optimFixedFF: bool = False
    distortion: Distortion = Distortion.NONE
    features:bool = False
    n_iter_max:int = 500
    dist_schedule : list = None

    def set_schedule(self,sch : list = None):
        if sch is None:
            if self.distortion== Distortion.NONE:
                self.dist_schedule = []
            elif self.distortion == Distortion.CONF_SCALE:
                self.dist_schedule = [x for x in np.logspace(3,-3,4)]
            elif self.distortion == Distortion.LSCM_B:
                self.dist_schedule = [x for x in np.logspace(-2,2,20)]
            elif self.distortion == Distortion.NORMAL:
                self.dist_schedule = [1.]
            elif self.distortion == Distortion.ID_B:
                self.dist_schedule = [x for x in np.logspace(-3,1,30)]
            else:
                self.dist_schedule = [x for x in np.logspace(2, -3, 6)]
        else:
            self.dist_schedule = sch
class Worker(M.Logger):
    """An abstract worker applying functions on a parametrization problem instance"""

    def __init__(self, 
        name : str, 
        instance : Instance, 
        options : Options = Options(),
        verbose_options : VerboseOptions = VerboseOptions()):
        
        self.instance : Instance = instance
        self.options : Options = options
        self.verbose_options : VerboseOptions = verbose_options
        M.Logger.__init__(self, name, self.verbose_options.logger_verbose)