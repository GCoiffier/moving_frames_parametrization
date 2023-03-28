import mouette as M
import mouette.geometry as geom

from .worker import *
from .instance import Instance
from .energy import *

import scipy.sparse as sp
import numpy as np
import os
from osqp import OSQP
from dataclasses import dataclass

from tqdm import tqdm, trange

@dataclass
class OptimHyperParameters:
    ### Hyper parameters for the Levenberg-Marquardt algorithm
    ENERGY_MIN : float = 1e-7
    MIN_STEP_NORM : float = 1e-4
    MIN_GRAD_NORM : float = 1e-6  # stopping criterion on projected gradient norm
    MIN_DELTA_E : float = 1e-4
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
        self.energies : list = None

        self.cstMat : sp.csc_matrix = None
        self.cstRHS_zeros : np.ndarray = None
        self.distMat : sp.csc_matrix = None
        self.distRHS_p : np.ndarray = None
        self.distRHS_m : np.ndarray = None


        self.stop_criterion_instance = None
        self.HP = optim_hp # hyperparameters

        self.det_threshold : float = 0.5 # minimal value for which the log barrier on jacobian determinants is 0
        self.sin_threshold : float = 0.99

        self.edge_weight : float = 1.
        self.FF_weight : float = 1.
        self.FFnorm_weight : float = 0.
        self.dist_weight : float = 1.

        self.lm_norm_t = 0.
        self.wMat : sp.csc_matrix = None

        self.n_snap = 0

    def compute_constraints(self):
        I = self.instance
        self.cstMat : sp.csc_matrix = None
        self.cstRHS_zeros : np.ndarray = None
        self.distMat : sp.csc_matrix = None
        self.distRHS : np.ndarray = None

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

        ### w on feature edges
        ncstr_w = 0
        # if not self.options.optimFixedFF:
        #     ncstr_w = len(I.feat.feature_edges) # for feature rotation constraints
        #     for e in I.feat.feature_edges:
        #         rows += [irow]
        #         cols += [I.var_sep_rot+e]
        #         coeffs += [1]
        #         irow += 1

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
        
        ### Finalize
        ncstr = ncstr_ff +  ncstr_w + ncstr_fe
        self.cstMat = sp.csc_matrix( (coeffs, (rows , cols)), shape=(ncstr,I.nvar))
        #self.cstRHS = self.cstMat.dot(I.var)
        self.cstRHS_zeros = np.zeros(ncstr)
        self.log(f"Number of constraints: {ncstr} ({ncstr_ff} + {ncstr_w} + {ncstr_fe})")

        ### Add distortion as linear constraints
        if self.options.distortion in [Distortion.LSCM_B, Distortion.ID_B, Distortion.ID_C]:
            coeffs, rows, cols = [], [], []
            if self.options.distortion == Distortion.LSCM_B:
                ncstr_dist = 2*len(I.mesh.faces)
                self.distRHS_p = self.dist_weight * np.ones(ncstr_dist)
                self.distRHS_m = - self.dist_weight * np.ones(ncstr_dist)
                for f in I.mesh.id_faces:
                    rows += [2*f, 2*f, 2*f+1, 2*f+1]
                    cols += [4*f, 4*f+3, 4*f+1, 4*f+2]
                    coeffs += [1, -1, 1, 1]
                    irow += 2
            elif self.options.distortion == Distortion.ID_B and self.dist_weight>0:
                ncstr_dist = 4*len(I.mesh.faces)
                self.distRHS_p = self.dist_weight * np.ones(ncstr_dist)
                self.distRHS_m = - self.dist_weight * np.ones(ncstr_dist)
                for f in I.mesh.id_faces:
                    rows += [4*f, 4*f+1, 4*f+2, 4*f+3]
                    cols += [4*f, 4*f+1, 4*f+2, 4*f+3]
                    coeffs += [1, 1, 1, 1]
                    self.distRHS_p[4*f] += 1 # for a and d interval is [1-x ; 1 +x]
                    self.distRHS_m[4*f] += 1
                    # self.distRHS_m[4*f+2] = 0
                    # self.distRHS_p[4*f+2] = 0
                    self.distRHS_p[4*f+3] += 1
                    self.distRHS_m[4*f+3] += 1
                    irow += 4
            elif self.options.distortion == Distortion.ID_C and self.dist_weight>0:
                ncstr_dist = 8*len(I.mesh.faces)
                self.distRHS_p = np.full(ncstr_dist, 1000)
                self.distRHS_m = - np.full(ncstr_dist, 1000)
                for f in I.mesh.id_faces:
                    for i in range(4):
                        rows += [8*f+2*i, 8*f+2*i, 8*f+2*i+1, 8*f+2*i+1]
                        cols += [4*f+i, I.var_sep_dist + f, 4*f+i, I.var_sep_dist + f]
                        coeffs += [1, 1, 1, -1]
                        self.distRHS_m[8*f+2*i] = 1. if i==0 or i==3 else 0.
                        self.distRHS_p[8*f+2*i+1] = 1. if i==0 or i==3 else 0.
                        irow += 2
            else:
                ncstr_dist = 0

            self.distMat = sp.csc_matrix( (coeffs, (rows , cols)), shape=(ncstr_dist,I.nvar))
            self.fullCstMat = sp.vstack([self.cstMat, self.distMat], format="csc")
            self.log(f"Number of distortion constraints: {ncstr_dist}")

    @property
    def distance_weight_matrix(self):
        if self.wMat is None:
            I = self.instance
            # lap = M.operators.laplacian_triangles(I.mesh,cotan=True,parallel_transport=False)

            wMatjac = sp.eye(4*len(I.mesh.faces), format="csc") #+ 0.5*sp.kron(lap, sp.eye(4, format="csc"))
            wMatRot = sp.eye(len(I.mesh.edges), format="csc")
            
            wMatFF = sp.eye(2*len(I.mesh.faces), format="csc") #+ 0.5*sp.kron(M.operators.laplacian_triangles(I.mesh,cotan=False,parallel_transport=False), sp.eye(2, format="csc"))
            # wMatFF = sp.kron(lap, sp.eye(2, format="csc"))

            if self.lm_norm_t > 0:
                t = self.lm_norm_t
                rows, cols, coeffs = [], [], []
                ## Conformal
                # for iT in I.mesh.id_faces:
                #     rows += [4*iT + k for k in [0,0,1,1,2,2,3,3]]
                #     cols += [4*iT + k for k in [0,3,1,2,1,2,0,3]]
                #     coeffs += [t, -t, t, t, t, t, -t, t]

                ## ARAP
                # for iT in I.mesh.id_faces:
                #     rows += [4*iT + k for k in [0,1,1,2,2,3]]
                #     cols += [4*iT + k for k in [2,1,3,2,0,1]]
                #     coeffs += [t, t2, t, t2, t, t]

                ## Identity
                # for iT in I.mesh.id_faces:
                #     rows += [4*iT + k for k in [1,2]]
                #     cols += [4*iT + k for k in [1,2]]
                #     coeffs += [t-1, t-1]

                ## Det
                for iT in I.mesh.id_faces:
                    rows += [4*iT + k for k in [0,0,1,1,2,2,3,3]]
                    cols += [4*iT + k for k in [0,3,1,2,1,2,0,3]]
                    coeffs += [1, t, 1, -t, -t, 1, t, 1]

                jac_dir_mat = sp.csc_matrix((coeffs, (rows, cols)), shape =(4*len(I.mesh.faces), 4*len(I.mesh.faces)))
                wMatjac += jac_dir_mat
            self.wMat = sp.block_diag([wMatjac, wMatRot, wMatFF], format="csc")
        return self.wMat

    def energy_noJ(self, X):
        I = self.instance
        F = []
        n = len(I.mesh.faces)
        D = self.options.distortion

        if self.options.optimFixedFF:
            F.append( constraint_edge_fixedFF_noJ(X, I.edge_indices, I.ref_edges, I.parallel_transport, I.var_sep_rot) )
        else:
            F.append(self.edge_weight*constraint_edge_noJ(X, I.edge_indices, I.ref_edges, I.parallel_transport, I.var_sep_rot))
            F.append(self.FF_weight*constraint_rotations_follows_ff_noJ(X, I.ff_indices, I.parallel_transport, I.var_sep_rot))
        
        if not self.options.optimFixedFF and self.FFnorm_weight>0:
            F.append( self.FFnorm_weight * ff_norm_noJ(X, n, I.var_sep_ff))

        # Barrier terms
        if self.det_threshold>0:
            F.append(barrier_det_noJ(X, n, self.det_threshold))
            #F.append(barrier_sin_noJ(X, n, self.det_threshold))
        
        # Distortion energies
        if self.dist_weight>0:
            if D == Distortion.LSCM:
                F.append(self.dist_weight * distortion_lscm_noJ(X, n))

            elif D == Distortion.ID : #or D==Distortion.ID_B:
                F.append(self.dist_weight * distortion_id_noJ(X,n))

            elif D == Distortion.ISO:
                F.append(self.dist_weight * distortion_isometric_noJ(X, n))

            elif D == Distortion.SHEAR:
                F.append(self.dist_weight * distortion_shear_noJ(X, n))
            
            elif D == Distortion.CONF_SCALE:
                F.append(self.dist_weight * conformal_connection_noJ(X, I.edge_indices, I.cotan_weight, I.var_sep_rot, I.var_sep_scale))
                # F.append(self.dist_weight * distortion_shear_noJ(X,n))
            
            elif D == Distortion.NORMAL:
                F.append(self.dist_weight * distortion_edge_normal_noJ(X, I.edge_indices, I.ref_edges, I.parallel_transport, I.var_sep_rot))
            
            elif D == Distortion.RIGID:
                F.append(self.dist_weight * distortion_rigid_noJ(X, I.edge_indices, I.rigid_ref_edges, I.rigid_matrices, I.var_sep_rot))

            elif D == Distortion.ID_C:
                F.append(self.dist_weight * distortion_norm2_noJ(X, n, I.var_sep_dist))
                #F.append(self.dist_weight * distortion_norm1_noJ(X, n, I.var_sep_dist))

        # F.append(0.1 * rot_norm_noJ(X, len(I.mesh.edges), I.var_sep_rot))

        return F

    def energy(self, X=None, jac=True):
        I = self.instance
        if X is None: X = I.var
        if not jac: return self.energy_noJ(X)

        names, f = [], []
        rowJ, colJ, vJ = [],[],[]
        off = 0
        n = len(I.mesh.faces)
        D = self.options.distortion 

        def add_energy(o, F,V,R,C):
            f.append(F)
            vJ.append(V)
            rowJ.append(R + o)
            colJ.append(C)
            return o + F.size

        names.append("Jac")
        if self.options.optimFixedFF:
            F,V,R,C = constraint_edge_fixedFF(X, I.edge_indices, I.ref_edges, I.parallel_transport, I.var_sep_rot)
            off = add_energy(off, F,V,R,C)

        else:
            F,V,R,C = constraint_edge(X, I.edge_indices, I.ref_edges, I.parallel_transport, I.var_sep_rot)
            off = add_energy(off, self.edge_weight*F, self.edge_weight*V, R, C)
            names.append("rotFF")
            F,V,R,C = constraint_rotations_follows_ff(X, I.ff_indices, I.parallel_transport, I.var_sep_rot)
            off = add_energy(off, self.FF_weight*F, self.FF_weight*V , R, C)
        
        if not self.options.optimFixedFF and self.FFnorm_weight>0:
            names.append("FFnorm")
            F,V,R,C = ff_norm(X,n,I.var_sep_ff)
            off = add_energy(off, self.FFnorm_weight * F, self.FFnorm_weight * V, R, C)

        # Barrier terms
        if self.det_threshold>0:
            # names.append("Sin")
            # F,V,R,C = barrier_sin(X, n, self.det_threshold)
            # off = add_energy(off, F,V,R,C)
            names.append("Det")
            F,V,R,C = barrier_det(X, n, self.det_threshold)
            off = add_energy(off, F,V,R,C)

        # Distortion energies
        if self.dist_weight>0 and D != Distortion.CONF_SCALE:
            if D == Distortion.LSCM:
                names.append("Lscm")
                F,V,R,C = distortion_lscm(X, n)
                off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)

            elif D == Distortion.ID : #or D==Distortion.ID_B:
                names.append("Id")
                F,V,R,C = distortion_id(X, n)
                off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)

            elif D == Distortion.ISO:
                names.append("Iso")
                F,V,R,C = distortion_isometric(X, n)
                off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)

            elif D == Distortion.SHEAR:
                names.append("Shear")
                F,V,R,C = distortion_shear(X, n)
                off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)
            
            elif D == Distortion.NORMAL:
                names.append("N")
                F,V,R,C = distortion_edge_normal(X, I.edge_indices, I.ref_edges, I.parallel_transport, I.var_sep_rot)
                off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)
            
            elif D == Distortion.RIGID:
                names.append("Rigid")
                F,V,R,C = distortion_rigid(X, I.edge_indices, I.rigid_ref_edges, I.rigid_matrices, I.var_sep_rot)
                off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)

            elif D == Distortion.ID_C:
                names.append("a")
                #F,V,R,C = distortion_norm2(X, n, I.var_sep_dist)
                F,V,R,C = distortion_norm1(X, n, I.var_sep_dist)
                off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)

        elif self.dist_weight>0 and D == Distortion.CONF_SCALE:
            names.append("Conf")
            F,V,R,C = conformal_connection(X, I.edge_indices, I.cotan_weight, I.var_sep_rot, I.var_sep_scale)
            off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)
            # names.append("Shear")
            # F,V,R,C = distortion_shear(X, n)
            # off = add_energy(off, self.dist_weight*F, self.dist_weight*V, R, C)
            
        # names.append("w")
        # F,V,R,C = rot_norm(X, len(I.mesh.edges), I.var_sep_rot)
        # off = add_energy(off, 0.1*F, 0.1*V, R, C)

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
                sp.eye(n,format=("csc")), q=-G, A = self.cstMat, l = self.cstRHS_zeros, u = self.cstRHS_zeros,
                verbose=self.verbose_options.qp_solver_verbose) #, linsys_solver='mkl pardiso')
        else:
            self.stop_criterion_instance.update(q=-G)
        x = self.stop_criterion_instance.solve().x
        #print(geom.norm(G), geom.norm(x), geom.distance(x,G))
        if x is None: return False, 0.
        xnorm = geom.norm(x)
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

        W = None # distance matrix (built when first needed)
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

                    if W is None : W = self.distance_weight_matrix
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
                #gamma = 2 * mu * Ex
                osqp_instance = OSQP()
                if self.options.distortion in [Distortion.LSCM_B, Distortion.ID_B, Distortion.ID_C] and self.dist_weight>0:
                    # compute RHS
                    rhs_m = np.concatenate([self.cstRHS_zeros, self.distRHS_m])
                    rhs_p = np.concatenate([self.cstRHS_zeros, self.distRHS_p])
                    rhs_xt = np.concatenate([self.cstRHS_zeros, self.distMat.dot(self.instance.var)])

                    osqp_instance.setup(JtJ + gamma*W, q=q, A=self.fullCstMat, l=rhs_m - rhs_xt, u= rhs_p - rhs_xt,
                        verbose=self.verbose_options.qp_solver_verbose, eps_abs=1e-5, eps_rel=1e-5,
                        max_iter=100, polish=True, check_termination=10, 
                        adaptive_rho=True, linsys_solver='mkl pardiso')
                else:
                    osqp_instance.setup(JtJ + gamma*W, q=q, A=self.cstMat, l=self.cstRHS_zeros, u=self.cstRHS_zeros,
                        verbose=self.verbose_options.qp_solver_verbose, 
                        eps_abs=1e-5, eps_rel=1e-5, alpha=1.0, 
                        max_iter=100, polish=False, check_termination=10, 
                        adaptive_rho=True, linsys_solver='mkl pardiso')
                s = osqp_instance.solve().x

                if s is not None:
                    ms = fx + Jx.dot(s)
                    ms = np.dot(ms,ms)/2 + gamma*np.dot(s,s)/2
                    fxs = self.energy(self.instance.var + s, jac=False)
                    fxs = np.concatenate(fxs)
                    Exs = np.dot(fxs,fxs)/2

                #rho = (Ex - Exs)/(Ex - ms)
                #update = (rho >= eta) # whether to make a step (True) or increase mu (False)
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

                if self.verbose_options.snapshot_freq and it%self.verbose_options.snapshot_freq==0:
                    self.make_snapshot()

                if self.verbose_options.log_freq>0 and it%self.verbose_options.log_freq==0:
                    energies = [np.dot(_f,_f)/2 for _f in f]
                    if self.verbose_options.tqdm:
                        tqdm_log = ""
                        for n,e in zip(names, energies):
                            tqdm_log+="{}: {:.2E} | ".format(n,e)
                        tqdm_log += "Total: {:.2E} | Grad: {:.2E} | ".format(Ex, grad_norm)
                        tqdm_log += "ΔE {:.2E} | Step {:.2E} | Mu: {:.2E}".format(RelDeltaE, step_norm, mu)
                        tqdm_log += " " * 10
                        tqdm.write(tqdm_log)
                    else:
                        log = ""
                        for n,e in zip(names, energies):
                            log+="{}: {:.2E} | ".format(n,e)
                        log += "Total: {:.3E} | Grad: {:.3E} |  ".format(np.sum(energies), grad_norm)
                        log += "ΔE {:.2E} | Step {:.2E} | Mu: {:.2E} ".format(RelDeltaE, step_norm, mu)
                        print(log)
                    
        except KeyboardInterrupt:
            self.log("Manual interruption")
        return self.end_optimization(Ex, "max iteration reached")

    def make_snapshot(self):
        I = self.instance
        snap_dir = os.path.join(self.verbose_options.output_dir, "snapshot")
        os.makedirs(snap_dir, exist_ok=True)
        ff = self.instance.export_frame_field()
        M.mesh.save(ff, f"{snap_dir}/FF_{self.n_snap}.geogram_ascii")

        out = M.mesh.copy(I.mesh)
        defect = out.vertices.create_attribute("defect", float)
        for v in I.mesh.id_vertices:
                angle = I.defect[v]
                for e in I.mesh.adjacency.vertex_to_edge(v):
                    u = I.mesh.adjacency.other_edge_end(e,v)
                    w = 2*atan(I.get_var_rot(e))
                    angle += w if v<u else -w
                defect[v] = angle

        det = out.faces.create_attribute("det", float)
        for f in I.mesh.id_faces:
            a,b,c,d = I.var[4*f:4*f+4]
            det[f] = a*d-b*c

        w = out.edges.create_attribute("w", float)
        for e in I.mesh.id_edges:
            w[e] = 2*atan(I.var[I.var_sep_rot+e])

        if self.options.distortion == Distortion.ID_C:
            a = out.faces.create_attribute("a", float)
            for f in I.mesh.id_faces:
                a[f] = I.var[I.var_sep_dist + f]

        M.mesh.save(out, f"{snap_dir}/snap_{self.n_snap}.geogram_ascii")

        flat = I.construct_param()
        M.mesh.save(flat,f"{snap_dir}/flat_{self.n_snap}.geogram_ascii")

        self.n_snap += 1

    def optimize(self):
        if not self.instance.initialized:
            self.log("Error : Variables are not initialized")
            raise Exception("Problem was not initialized.")

        for step, weight in enumerate(self.options.dist_schedule):
            self.dist_weight = weight
            self.compute_constraints()
            self.log(f"Distortion weight: {self.dist_weight:.2E}\n\n")
            self.LevenbergMarquardt(self.options.n_iter_max)
            # self.make_snapshot()
            if self.verbose_options.optim_verbose:
                print()
        
        # Last pass without distortion
        self.dist_weight = 0.
        #self.FF_weight = 1.
        self.compute_constraints()
        self.log(f"Distortion weight: {self.dist_weight:.2E}\n\n")
        energy = self.LevenbergMarquardt(self.options.n_iter_max)
        if self.verbose_options.optim_verbose: print()
        return energy