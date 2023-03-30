import numpy as np
import os
from osqp import OSQP
import scipy.sparse as sp
from dataclasses import dataclass

import mouette as M
import mouette.geometry as geom

from .instance import Instance
from .energy import *
from .worker import *
from .common import *

from tqdm import tqdm, trange
from tqdm.utils import _term_move_up
prefix = _term_move_up() + '\r'

@dataclass
class OptimHyperParameters:
    ### Hyper parameters for the Levenberg-Marquardt algorithm
    ENERGY_MIN : float = 1e-7 # stopping criterion on the energy value
    MIN_DELTA_E : float = 1e-4 # stopping criterion on the energy difference value
    MIN_STEP_NORM : float = 1e-4  # stopping criterion on the step vector (x_n+1 - x_n) norm
    MIN_GRAD_NORM : float = 1e-5  # stopping criterion on projected gradient norm
    MU_MAX : float = 1e8 # stopping criterion on mu value
    MU_MIN : float = 1e-8
    alpha : float = 0.5 # if iteration is a success, mu = alpha * mu
    beta : float = 2. # if iteration fails, mu = beta * mu

class Optimizer(Worker):

    def __init__(self, 
        instance: Instance, 
        options = Options(),
        verbose_options = VerboseOptions(),
        optim_hp = OptimHyperParameters()):
        super().__init__("Optimizer", instance, options, verbose_options)

        self.cstMat : sp.csc_matrix = None
        self.cstRHS : np.ndarray = None # l for lower

        self.stop_criterion_instance = None

        self.HP = optim_hp # hyperparameters

        self.dist_weight : float = 1.
        self.dist_area_balance : float = 0.1

        self.edge_weight : float = 1.
        self.FF_weight : float = 10.
        self.singu_det_threshold = 0.5 # minimal ratio value for the singularity barrier term 
        self.orient_det_threshold = 0.5 # minimal ratio value for the orientation barrier term

    def compute_constraints(self):
        I = self.instance

        coeffs, rows, cols = [], [], []
        self.cstRHS, self.cstRHS_u = [], []
        irow = 0

        if self.options.free_boundary:
            # lock only one frame
            ncstr_ff = 2
            rows += [irow, irow+1]
            cols += [I.var_sep_ff, I.var_sep_ff + 1]
            coeffs += [1, 1]
            irow+=2
            # other constraints are ignored
            ncstr_fe = 0
            ncstr_ring_fe = 0

        else:

            ### Lock frames on features and boundary
            ncstr_ff = 0
            if not self.options.optimFixedFF:
                # if optim fixed ff, frames will not change so constraints are useless
                ncstr_ff = 2*len(I.feat.feature_vertices)
                for v in I.feat.feature_vertices:
                    rows += [irow, irow+1]
                    cols += [I.var_sep_ff + 2*v, I.var_sep_ff + 2*v+1]
                    coeffs += [1, 1]
                    irow+=2
                self.cstRHS += [0]*ncstr_ff
                self.cstRHS_u += [0]*ncstr_ff

            ### Charts along features should align with feature normal
            ncstr_fe = 0
            ncstr_fe = sum([ len(I.feat.local_feat_edges[v]) for v in I.feat.feature_vertices]) # for border and feature curves alignment with normal
            self.cstRHS += [0]*ncstr_fe
            self.cstRHS_u += [0]*ncstr_fe
            for v in I.feat.feature_vertices:
                for ie in I.feat.local_feat_edges[v]:
                    e = I.work_mesh.connectivity.vertex_to_edge(v)[ie]
                    direct = I.work_mesh.edges[e][0]==v
                    nmid = 4*e if direct else 4*e + 2
                    mid = M.Vec(I.var[nmid], I.var[nmid + 1])
                    n = M.Vec.normalized(M.Vec(-mid.y, mid.x))
                    # dot(e, n) = 0
                    rows += [irow, irow]
                    cols += [nmid, nmid+1]
                    coeffs += [n.x, n.y]
                    irow += 1

            ### split rings over feature edges
            ncstr_ring_fe = 0 # for rings split by feature edges
            for v,split_v in I.feat.cut_vertices.items():
                if len(split_v)<2 : continue
                if I.mesh.is_vertex_on_border(v):
                    ncstr_ring_fe += 2*( len(split_v) - 1)
                else:
                    ncstr_ring_fe += 2*len(split_v)
            self.cstRHS += [0]*ncstr_ring_fe
            self.cstRHS_u += [0]*ncstr_ring_fe
                    
            for v,split_v in I.feat.cut_vertices.items():
                if len(split_v)<2 : continue # no need to split (should not happen)
                for a,b in M.utils.consecutive_pairs(split_v):
                    # extract pair of edges
                    va,vb = I.work_mesh.connectivity.vertex_to_vertex(a)[-1], I.work_mesh.connectivity.vertex_to_vertex(b)[0]
                    iea, ieb = I.work_mesh.connectivity.edge_id(a,va), I.work_mesh.connectivity.edge_id(b,vb)
                    iea = 4*iea if a<va else 4*iea+2
                    ieb = 4*ieb if b<vb else 4*ieb+2
                    Aab, Aba = -I.connection.transport(a,va), -I.connection.transport(b,vb)
                    # ea * exp( - i A_ab) - eb * exp( - i A_ba ) = 0
                    rows += [irow, irow, irow, irow]
                    cols += [iea, iea+1, ieb, ieb+1]
                    coeffs += [cos(Aab), -sin(Aab), -cos(Aba), sin(Aba) ]
                    rows += [irow+1, irow+1, irow+1, irow+1]
                    cols += [iea, iea+1, ieb, ieb+1]
                    coeffs += [ sin(Aab), cos(Aab), -sin(Aba), -cos(Aba) ]
                    irow += 2
                
                if not I.mesh.is_vertex_on_border(v):
                    # for the last pair (closure of ring), we apply the defect:
                    # ea * exp( - i A_ab) = eb * exp( - i A_ba ) * exp( i * defect)
                    a,b = split_v[-1], split_v[0]
                    va,vb = I.work_mesh.connectivity.vertex_to_vertex(a)[-1], I.work_mesh.connectivity.vertex_to_vertex(b)[0]
                    iea, ieb = I.work_mesh.connectivity.edge_id(a,va), I.work_mesh.connectivity.edge_id(b,vb)
                    iea = 4*iea if a<va else 4*iea+2
                    ieb = 4*ieb if b<vb else 4*ieb+2
                    Aab, Aba = -I.connection.transport(a,va), -I.connection.transport(b,vb) + 2*pi/self.instance.order*(self.instance.order-I.feat.corners_no_cuts[v])
                    rows += [irow, irow, irow, irow]
                    cols += [iea, iea+1, ieb, ieb+1]
                    coeffs += [ cos(Aab), -sin(Aab), -cos(Aba), sin(Aba)]
                    rows += [irow+1, irow+1, irow+1, irow+1]
                    cols += [iea, iea+1, ieb, ieb+1]
                    coeffs += [ sin(Aab), cos(Aab), -sin(Aba), -cos(Aba)]
                    irow += 2

        ### Finalize
        ncstr = ncstr_ff + ncstr_fe + ncstr_ring_fe
        self.cstMat = sp.coo_matrix( (coeffs, (rows , cols)), shape=(ncstr,I.nvar)).tocsc()
        self.cstRHS = np.zeros(ncstr)
        self.log(f"Number of constraints: {ncstr} ({ncstr_ff} + {ncstr_fe} + {ncstr_ring_fe})")

    def energy_noJ(self, X):
        I = self.instance
        F = []

        n = len(I.work_mesh.edges)
        m = 3*len(I.work_mesh.faces)
        if self.options.optimFixedFF:
            F.append(constraint_edge_fixed_w_noJ1(X, I.PT_array, n, m, I.var_sep_rot))
            F.append(constraint_edge_fixed_w_noJ2(X, I.edge_indices, I.PT_array, n, m, I.var_sep_rot))
        else:
            F.append(self.edge_weight*constraint_edge_noJ1(X, I.edge_lengths, I.PT_array, n, m, I.var_sep_rot))
            F.append(self.edge_weight*constraint_edge_noJ2(X, I.edge_indices, I.edge_lengths, I.PT_array, n, m, I.var_sep_rot))

            if self.options.free_boundary:
                F.append(self.FF_weight*constraint_rotations_follow_ff_order1_noJ(X, I.rotFF_indices, I.PT_array, n))
            else:
                F.append(self.FF_weight*constraint_rotations_follow_ff_noJ(X, I.rotFF_indices, I.PT_array, n, I.order))
        
        # Barrier terms
        if self.options.free_boundary:
            F.append(barrier_det_corner_noJ(X, I.quad_indices, I.ref_dets, self.orient_det_threshold, self.singu_det_threshold))
        else:
            F.append(barrier_det_full_noJ(X, I.quad_indices, I.ref_dets, self.orient_det_threshold, self.singu_det_threshold))

        # Distortion energies
        if self.dist_weight>0:
            if self.options.distortion == Distortion.LSCM:
                F.append(self.dist_weight * distortion_lscm_noJ(X, I.quad_indices, I.init_var))

            elif self.options.distortion == Distortion.ARAP:
                F.append(self.dist_weight * distortion_isometric_noJ(X, I.quad_indices, I.dist_matrices))
            
            elif self.options.distortion == Distortion.AREA:
                F.append(self.dist_weight * distortion_det_noJ(X, I.quad_indices, I.ref_dets))
                F.append(self.dist_weight * self.dist_area_balance * distortion_lscm_noJ(X, I.quad_indices, I.init_var))
        return F

    def energy(self, X=None, jac=True):
        I = self.instance
        if X is None: X = I.var
        if not jac: return self.energy_noJ(X)

        names, f = [], []
        rowJ, colJ, vJ = [],[],[]
        off = 0

        def add_energy(o, F,V,R,C):
            f.append(F)
            vJ.append(V)
            rowJ.append(R + o)
            colJ.append(C)
            return o + F.size

        n = len(I.work_mesh.edges)
        m = 3*len(I.work_mesh.faces)
        if self.options.optimFixedFF:
            names.append("E1")
            F,V,R,C = constraint_edge_fixed_w1(X, I.PT_array, n,m, I.var_sep_rot)
            off = add_energy(off, F,V,R,C)
            names.append("E2")
            F,V,R,C = constraint_edge_fixed_w2(X, I.edge_indices, I.PT_array, n,m, I.var_sep_rot)
            off = add_energy(off, F,V,R,C)

        else:
            names.append("E1")
            F,V,R,C = constraint_edge1(X, I.edge_lengths, I.PT_array, n,m, I.var_sep_rot)
            off = add_energy(off, self.edge_weight*F, self.edge_weight*V, R, C)
            names.append("E2")
            F,V,R,C = constraint_edge2(X, I.edge_indices, I.edge_lengths, I.PT_array, n,m, I.var_sep_rot)
            off = add_energy(off, self.edge_weight*F, self.edge_weight*V, R, C)

            names.append("rotFF")
            if self.options.free_boundary:
                F,V,R,C = constraint_rotations_follow_ff_order1(X, I.rotFF_indices, I.PT_array, n)
            else:
                F,V,R,C = constraint_rotations_follow_ff(X, I.rotFF_indices, I.PT_array, n, I.order)
            off = add_energy(off, self.FF_weight*F, self.FF_weight*V , R, C)
        
        # Barrier terms
        names.append("Det")
        if self.options.free_boundary:
            # If free boundary mode, singularities cannot appear so their position is not relevant to bound
            F,V,R,C = barrier_det_corner(X, I.quad_indices, I.ref_dets, self.orient_det_threshold, self.singu_det_threshold)
        else:
            F,V,R,C = barrier_det_full(X, I.quad_indices, I.ref_dets, self.orient_det_threshold, self.singu_det_threshold)
        off = add_energy(off, F,V,R,C)

        # Distortion energies
        if self.dist_weight>0:
            if self.options.distortion == Distortion.LSCM:
                names.append("Lscm")
                F,V,R,C = distortion_lscm(X, I.quad_indices, I.init_var)

            elif self.options.distortion == Distortion.ARAP:
                names.append("Arap")
                F,V,R,C = distortion_isometric(X, I.quad_indices, I.dist_matrices)

            elif self.options.distortion == Distortion.AREA:
                names.append("Scale")
                F,V,R,C = distortion_det(X, I.quad_indices, I.ref_dets)
                off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)

                names.append("Lscm")
                F,V,R,C = distortion_lscm(X, I.quad_indices, I.init_var)
                F *= self.dist_area_balance
                V *= self.dist_area_balance

            off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)

        rowJ,colJ,vJ = np.concatenate(rowJ), np.concatenate(colJ), np.concatenate(vJ)
        J = sp.coo_matrix( (vJ, (rowJ,colJ)), shape=(off, self.instance.nvar))
        return names, f, J.tocsc()

    def stop_criterion(self, G, tol):
        """
        projected gradient on the orthogonal of space spanned by constraints should have small norm
        """
        n = self.cstMat.shape[1]
        if self.stop_criterion_instance is None:
            self.stop_criterion_instance = OSQP()
            self.stop_criterion_instance.setup(
                sp.eye(n,format=("csc")), q=-2*G, A = self.cstMat, l = self.cstRHS, u = self.cstRHS,
                verbose=self.verbose_options.qp_solver_verbose) #, linsys_solver='mkl pardiso')
        else:
            self.stop_criterion_instance.update(q=-2*G)
        x = self.stop_criterion_instance.solve().x
        if x is None: return False, 0.
        xnorm = np.sqrt(np.dot(x,x))
        # xnorm = np.sqrt(np.dot(G,G))
        return xnorm<tol, xnorm

    def end_optimization(self, energy, message):
        if self.verbose_options.optim_verbose:
            print("\n")
            self.log("End of optimization: "+message)
            self.log(f"Final Energy: {energy:.4E}")
            print("\n\r")
        return energy

    def LevenbergMarquardt(self, n_iter_max):
        if n_iter_max <= 0 : return

        mu = 1.
        mu_avg = mu
        update = True
        grad_norm = None
        if self.verbose_options.optim_verbose and self.verbose_options.tqdm:
            iterobj = trange(n_iter_max, total=n_iter_max, position=1, leave=False, ncols=100, unit="")
        else:
            iterobj = range(n_iter_max)

        Id = None # identity matrix (built when first needed)
        step_norm = 0.
        RelDeltaE = 0.
        
        try:
            # Levenberg-Marquardt optimization
            for it in iterobj:                

                if update: # energy only changes if we performed a step last iteration
                    names, f, Jx = self.energy()
                    fx = np.concatenate(f)
                    
                    Jt = Jx.transpose()
                    JtJ = Jt.dot(Jx)
                    q = Jt.dot(fx) # gradient

                    if Id is None : Id = sp.identity(JtJ.shape[0])
                    Ex = np.dot(fx,fx)/2

                    if Ex<self.HP.ENERGY_MIN:
                        return self.end_optimization(Ex, "Ex < Ex_min") # zero found
                    if np.isnan(Ex):
                        return self.end_optimization(Ex, "NaN value in energy")
                    if np.isposinf(Ex):
                        mu = self.HP.beta*mu
                        update=False
                        continue
                    grad_stop, grad_norm = self.stop_criterion(q, self.HP.MIN_GRAD_NORM)
                    if grad_stop:
                        return self.end_optimization(Ex, "projected grad norm on constraints < min_grad_norm")

                if mu>self.HP.MU_MAX:
                    return self.end_optimization(Ex, "mu > mu_max")
                    
                gamma = mu * np.sqrt(2*Ex)
                osqp_instance = OSQP()
                osqp_instance.setup(JtJ + gamma*Id, q=q, A=self.cstMat, l=self.cstRHS, u=self.cstRHS,
                    verbose=self.verbose_options.qp_solver_verbose,
                    eps_abs=1e-3, eps_rel=1e-3,
                    max_iter=100, polish=True, check_termination=10, 
                    adaptive_rho=True, linsys_solver='mkl pardiso')
                s = osqp_instance.solve().x

                if s[0] is not None:
                    ms = fx + Jx.dot(s)
                    ms = np.dot(ms,ms)/2 + gamma*np.dot(s,s)/2
                    fxs = self.energy(self.instance.var + s, jac=False)
                    fxs = np.concatenate(fxs)
                    Exs = np.dot(fxs,fxs)/2

                update = (s is not None) and (Exs <= ms)  # whether to make a step (True) or increase mu (False)
                if update:
                    RelDeltaE = abs(Exs-Ex)/Ex
                    if RelDeltaE < self.HP.MIN_DELTA_E : #and it>10:
                        return self.end_optimization(Ex, "Relative progression of energy < ΔE_min")
                    self.instance.var += s
                    step_norm = geom.norm(s)
                    if step_norm < self.HP.MIN_STEP_NORM : #and it>10:
                        return self.end_optimization(Ex, "Step norm < min_step_norm")
                    mu, mu_avg = max(self.HP.MU_MIN, self.HP.alpha*mu_avg), mu
                else:
                    mu = self.HP.beta*mu

                if self.verbose_options.optim_verbose and self.verbose_options.log_freq>0 and it%self.verbose_options.log_freq==0:
                    energies = [np.dot(_f,_f)/2 for _f in f]
                    if self.verbose_options.tqdm:
                        tqdm_log = prefix
                        for n,e in zip(names, energies):
                            tqdm_log+="{}: {:.2E} | ".format(n,e)
                        tqdm_log += "Total: {:.2E} | Grad: {:.2E} | ".format(Ex, grad_norm)
                        tqdm_log += "ΔE {:.2E} | Step {:.2E} | Mu: {:.2E}".format(RelDeltaE, step_norm, mu)
                        tqdm_log += " " * 10
                        tqdm.write(prefix)
                        tqdm.write(tqdm_log)
                    else:
                        log = f"{it+1}/{n_iter_max} | "
                        for n,e in zip(names, energies):
                            log+="{}: {:.2E} | ".format(n,e)
                        log += "Total: {:.2E} | Grad: {:.2E} | ".format(Ex, grad_norm)
                        log += "ΔE {:.2E} | Step {:.2E} | Mu: {:.2E}".format(RelDeltaE, step_norm, mu)
                        print(log)
                    
        except KeyboardInterrupt:
            self.log("Manual interruption")
        # except ValueError:
        #     self.log("Manual interruption")
        return self.end_optimization(Ex, "max iteration reached")

    def optimize(self):
        if not self.instance.initialized:
            self.log("Error : Variables are not initialized")
            raise Exception("Problem was not initialized.")

        for step, weight in enumerate(self.options.dist_schedule):
            self.dist_weight = weight
            self.compute_constraints()
            self.log(f"Distortion weight: {self.dist_weight:.2E}")
            energy = self.LevenbergMarquardt(self.options.n_iter_max)
            if self.verbose_options.optim_verbose: print()
        
        # Last pass without distortion
        self.dist_weight = 0
        self.compute_constraints()
        self.log(f"Distortion weight: {self.dist_weight:.2E}")
        energy = self.LevenbergMarquardt(self.options.n_iter_max)
        if self.verbose_options.optim_verbose: print()
        return energy
