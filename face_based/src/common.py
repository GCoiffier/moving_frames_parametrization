from enum import Enum
from dataclasses import dataclass
from math import log
import numpy as np
from numba import jit
import csv
import numpy as np

##### Distortion #####

class Distortion(Enum):
    """Description of the different energies used a a distortion in the optimization.
    """
    NONE = 0

    LSCM = 1    # conformal distortion as energy
    LSCM_M = 2  # conformal distortion as change of metric in optimizer

    ARAP = 4    # isometric distortion as energy
    ARAP_M = 5  # isometric distortion as change of metric in optimizer

    ID = 6      # identity distortion as energy
    ID_M = 7    # identity distortion as change of metric in optimizer
    ID_cst = 8  # identity distortion as linear constraints

    AREA = 9    # area distortion as energy
    AREA_M = 10 # area distortion as change of metric in optimizer

    @staticmethod
    def from_string(s:str):
        arg = s.lower()

        if "lscm_metric" in arg:
            return Distortion.LSCM_M
        if "lscm" in arg:
            return Distortion.LSCM
        
        if "arap_metric" in arg:
            return Distortion.ARAP_M
        if "arap" in arg:
            return Distortion.ARAP

        if "id_metric" in arg:
            return Distortion.ID_M
        if "id_cst" in arg:
            return Distortion.ID_cst
        if "id" in arg:
            return Distortion.ID

        if "area_metric" in arg:
            return Distortion.AREA_M
        if "area" in arg:
            return Distortion.AREA

        return Distortion.NONE


##### Default running options #####

@dataclass
class VerboseOptions:
    output_dir : str = ""
    logger_verbose : bool = True
    qp_solver_verbose : bool = False
    optim_verbose : bool = True
    log_freq : int = 0
    tqdm : bool = True

@dataclass
class Options:
    initSmooth: bool = False
    optimFixedFF: bool = False
    distortion: Distortion = Distortion.NONE
    features:bool = False
    n_iter_max:int = 500
    dist_schedule : list = None

    def set_schedule(self,sch : list = None):
        if sch is None:
            self.dist_schedule = {
                Distortion.NONE : [],
                
                Distortion.LSCM : [x for x in np.logspace(2, -3, 6)],
                Distortion.ARAP : [x for x in np.logspace(2, -3, 6)],
                Distortion.ID   : [x for x in np.logspace(2, -3, 6)],
                Distortion.AREA : [x for x in np.logspace(2, -3, 6)],

                Distortion.LSCM_M : [],
                Distortion.ARAP_M : [],
                Distortion.ID_M   : [],
                Distortion.AREA_M : [],
                
                Distortion.ID_cst : [x for x in np.logspace(-3,1,30)],
            }[self.distortion]
        else:
            self.dist_schedule = sch

######### Miscallenous utility functions #########

@jit(nopython=True, fastmath=True)
def barrier(x,t):
    if x<= 0 : return 1E9
    if x>t: return 0
    return log(x/t)**2

@jit(nopython=True, fastmath=True)
def barrier_prime(x, t):
    if x<0 : return 1E9
    if x>t : return 0
    return 2*log(x/t)/x

@jit(nopython=True)
def complex_to_mat(c : complex) -> np.ndarray:
    return np.array([[c.real, c.imag], [ -c.imag, c.real]], dtype=np.float64)

def export_dict_as_csv(data : dict, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)