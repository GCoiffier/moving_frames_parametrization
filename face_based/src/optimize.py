import mouette as M

from .worker import *
from .instance import Instance
from .energy import *

import scipy.sparse as sp
import numpy as np

from osqp import OSQP

##########

def get_osqp_lin_solver():
    try:
        inst = OSQP()
        inst.setup(P=sp.identity(1, format="csc"), verbose=False, linsys_solver="mkl pardiso")
        return "mkl pardiso"
    except ValueError:
        return "qdldl"

##########


class Optimizer(Worker):
    def __init__(self, 
    instance: Instance, 
    options = Options(), 
    verbose_options = VerboseOptions()):
        super().__init__("Optimizer", instance, options, verbose_options)
        self.energies : list = None

        #### Linear constraints matrix and bounds
        self.cstMat   : sp.csc_matrix = None
        self.cstRHS_l : np.ndarray    = None
        self.cstRHS_u : np.ndarray    = None # lower and upper bounds

        #### Metric matrix
        self.metric_t = 0.95
        self._metric_mat : sp.csc_matrix = None

        self.optimizer : M.optimize.LevenbergMarquardt = None

        #### Optimizer weights
        self.det_threshold : float = 0.5 # minimal value for which the log barrier on jacobian determinants is 0
        self.edge_weight : float = 10.
        self.FF_weight : float = 1.
        self.dist_weight : float = 1.

        self._linsys_solver = get_osqp_lin_solver()
        if self._linsys_solver == 'qdldl':
            self.log("WARNING : OSQP will run with qdldl as its internal linear solver.\n For better performance, we recommend to install and use 'mkl pardiso' instead : https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html")
            print("\n\n")

    def compute_constraints(self):
        I = self.instance
        self.cstMat : sp.csc_matrix = None
        self.cstRHS_zeros : np.ndarray = None

        coeffs = []
        rows = []
        cols = []
        irow = 0

        ### frame field
        ncstr_ff = 0
        if not self.options.optimFixedFF:
            if len(I.feat.feature_vertices)==0:
                # fix a random frame
                ncstr_ff = 2
                rows += [irow, irow+1]
                cols += [I.var_sep_ff, I.var_sep_ff +1] # frame 0
                coeffs += [1, 1]
                irow += 2
            else:
                ncstr_ff = 0 # for feature frame field constraints
                for e in I.feat.feature_edges:
                    for T in I.mesh.half_edges.edge_to_triangles(*I.mesh.edges[e]):
                        if T is None: continue # edge may be on boundary
                        rows += [irow, irow+1]
                        cols += [I.var_sep_ff + 2*T, I.var_sep_ff + 2*T+1]
                        coeffs += [1, 1]
                        irow += 2
                        ncstr_ff += 2

        ### feature edges -> align with normal
        ncstr_fe = 0 # for border and feature curves alignment with normal
        for ie in I.feat.feature_edges:
            u,v = I.mesh.edges[ie]
            for T in I.mesh.half_edges.edge_to_triangles(u,v):
                if T is None: continue
                X,Y = I.local_base(T)
                E = I.mesh.vertices[v] - I.mesh.vertices[u]
                e = M.Vec(X.dot(E), Y.dot(E))
                e.normalize()
                rows += [irow, irow, irow, irow]
                cols += [4*T, 4*T+1, 4*T+2, 4*T+3]
                coeffs += [-e.x*e.y, e.x*e.x, -e.y*e.y, e.x*e.y]
                irow += 1
                ncstr_fe += 1
        
        ### Finalize without distortion
        self.cstMat = sp.csc_matrix( (coeffs, (rows , cols)), shape=(ncstr_ff + ncstr_fe,I.nvar))
        self.cstRHS_l = self.cstMat.dot(I.var)
        self.cstRHS_u = self.cstMat.dot(I.var)

        ### Add distortion as linear constraints
        ncstr_dist = 0
        if self.options.distortion == Distortion.ID_cst and self.dist_weight>0:
            coeffs, rows, cols = [], [], []
            ncstr_dist = 4*len(I.mesh.faces)
            distRHS_u = self.dist_weight * np.ones(ncstr_dist)
            distRHS_l = - self.dist_weight * np.ones(ncstr_dist)
            for f in I.mesh.id_faces:
                rows += [4*f, 4*f+1, 4*f+2, 4*f+3]
                cols += [4*f, 4*f+1, 4*f+2, 4*f+3]
                coeffs += [1, 1, 1, 1]
                distRHS_u[4*f] += 1 # for a and d interval is [1-x ; 1+x]
                distRHS_l[4*f] += 1
                distRHS_u[4*f+3] += 1
                distRHS_l[4*f+3] += 1
                # distRHS_m[4*f+2] = 0
                # distRHS_p[4*f+2] = 0 # forces jacobians to be upper triangular

            distMat = sp.csc_matrix( (coeffs, (rows , cols)), shape=(ncstr_dist,I.nvar))
            self.cstMat = sp.vstack([self.cstMat, distMat], format="csc")
            self.cstRHS_u = np.concatenate([self.cstRHS_u, distRHS_u])
            self.cstRHS_l = np.concatenate([self.cstRHS_l, distRHS_l])
                    
        ncstr = ncstr_ff + ncstr_fe + ncstr_dist
        self.log(f"Number of constraints: {ncstr} ({ncstr_ff} + {ncstr_fe} + {ncstr_dist})")

    @property
    def metric_matrix(self):
        if self._metric_mat is None:
            I = self.instance

            ###### Metric matrix for jacobians ######
            wMatjac = sp.eye(4*len(I.mesh.faces), format="csc")
            t = self.metric_t
            rows, cols, coeffs = [], [], []
            if self.options.distortion == Distortion.LSCM_M:
                ## Conformal
                for iT in I.mesh.id_faces:
                    rows += [4*iT + k for k in [0,0,1,1,2,2,3,3]]
                    cols += [4*iT + k for k in [0,3,1,2,1,2,0,3]]
                    coeffs += [t, -t, t, t, t, t, -t, t]

            elif self.options.distortion == Distortion.ARAP_M:
                ## ARAP
                for iT in I.mesh.id_faces:
                    rows += [4*iT + k for k in [0,1,1,2,2,3]]
                    cols += [4*iT + k for k in [2,1,3,2,0,1]]
                    coeffs += [t, 0.5*t, t, 0.5*t, t, t]

            elif self.options.distortion == Distortion.ID_M:
                ## Identity
                for iT in I.mesh.id_faces:
                    rows += [4*iT + k for k in [1,2]]
                    cols += [4*iT + k for k in [1,2]]
                    coeffs += [t-1, t-1]

            elif self.options.distortion == Distortion.AREA_M:
                ## Area
                for iT in I.mesh.id_faces:
                    rows += [4*iT + k for k in [0,0,1,1,2,2,3,3]]
                    cols += [4*iT + k for k in [0,3,1,2,1,2,0,3]]
                    coeffs += [1, t, 1, -t, -t, 1, t, 1]

            jac_dir_mat = sp.csc_matrix((coeffs, (rows, cols)), shape =(4*len(I.mesh.faces), 4*len(I.mesh.faces)))
            wMatjac += jac_dir_mat

            ###### Metric matrix for FF and rot ######
            wMatRot = sp.eye(len(I.mesh.edges), format="csc")
            wMatFF  = sp.eye(2*len(I.mesh.faces), format="csc")

            ###### Finalize ######
            self._metric_mat = sp.block_diag([wMatjac, wMatRot, wMatFF], format="csc")
        return self._metric_mat


    def prepare_optimizer(self):
        I = self.instance
        nvar = I.var.size
        n = len(I.mesh.faces)
        D = self.options.distortion 

        def prepare_fun(fun,*args):
            def aux(X):
                F,V,R,C = fun(X,*args)
                return F, sp.csc_matrix((V,(R,C)), shape=(F.size, nvar))
            return aux
        
        self.optimizer = M.optimize.LevenbergMarquardt(lin_solver=self._linsys_solver)
        self.optimizer.HP.N_ITER_MAX = self.options.n_iter_max
        self.optimizer.register_constraints(self.cstMat, self.cstRHS_l, self.cstRHS_u)
        self.optimizer.set_metric_matrix(self.metric_matrix)

        if self.options.optimFixedFF:
            self.optimizer.register_function(
                prepare_fun(constraint_edge_fixedFF, I.edge_indices, I.ref_edges, I.PT_array, I.var_sep_rot),
                lambda X : constraint_edge_fixedFF_noJ(X, I.edge_indices, I.ref_edges, I.PT_array, I.var_sep_rot), 
                self.edge_weight, "Jac"
            )
        else:
            self.optimizer.register_function(
                prepare_fun(constraint_edge, I.edge_indices, I.ref_edges, I.PT_array, I.var_sep_rot),
                lambda X : constraint_edge_noJ(X, I.edge_indices, I.ref_edges, I.PT_array, I.var_sep_rot),
                self.edge_weight, "Jac"
            )

            self.optimizer.register_function(
                prepare_fun(constraint_rotations_follows_ff, I.ff_indices, I.PT_array, I.var_sep_rot),
                lambda X : constraint_rotations_follows_ff_noJ(X, I.ff_indices, I.PT_array, I.var_sep_rot),
                self.FF_weight, "FF"
            )

        if self.det_threshold>0:
            self.optimizer.register_function(
                prepare_fun(barrier_det, n, self.det_threshold),
                lambda X : barrier_det_noJ(X, n, self.det_threshold),
                1., "Det"
            )
        
        if D == Distortion.LSCM:
            self.optimizer.register_function(
                prepare_fun(distortion_lscm, n),
                lambda X : distortion_lscm_noJ(X,n),
                self.dist_weight, "LSCM"
            )
        elif D == Distortion.ARAP:
            self.optimizer.register_function(
                prepare_fun(distortion_isometric, n),
                lambda X : distortion_isometric_noJ(X,n),
                self.dist_weight, "ARAP"
            )
        elif D == Distortion.AREA:
            self.optimizer.register_function(
                prepare_fun(distortion_det , n),
                lambda X : distortion_det_noJ(X,n),
                self.dist_weight, "area"
            )
            self.optimizer.register_function(
                prepare_fun(distortion_lscm, n),
                lambda X : distortion_lscm_noJ(X,n),
                0.1*self.dist_weight, "LSCM"
            )
        elif D == Distortion.ID:
            self.optimizer.register_function(
                prepare_fun(distortion_id, n),
                lambda X : distortion_id_noJ(X,n),
                self.dist_weight, "Id"
            )


    def optimize(self):
        if not self.instance.initialized:
            self.log("Error : Variables are not initialized")
            raise Exception("Problem was not initialized.")

        for weight in self.options.dist_schedule:
            self.dist_weight = weight
            self.compute_constraints()
            self.log(f"Distortion weight: {self.dist_weight:.2E}\n\n")
            self.prepare_optimizer()
            self.optimizer.run(self.instance.var)
            self.instance.var = self.optimizer.X
            if self.verbose_options.optim_verbose:
                print()
        
        # Last pass without distortion
        self.dist_weight = 0.
        self.compute_constraints()
        self.log(f"Distortion weight: {self.dist_weight:.2E}\n\n")
        self.prepare_optimizer()
        energy = self.optimizer.run(self.instance.var)
        self.instance.var = self.optimizer.X
        if self.verbose_options.optim_verbose: print()
        return energy