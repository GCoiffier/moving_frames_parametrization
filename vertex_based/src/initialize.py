import math
import numpy as np
import scipy.sparse as sp
from osqp import OSQP

import mouette as M
import mouette.geometry as geom

from .featurecut import FeatureCutter
from .instance import Instance
from .worker import *
from .common import *

##### Initialization functions #####

def rescale_input_mesh(I : Instance):
    """Rescales the mesh to avoid numerical problems"""
    Lmean = M.attributes.mean_edge_length(I.mesh)
    I.mesh = M.processing.split_double_boundary_edges_triangles(I.mesh) # sanity check
    I.mesh = M.transform.scale(I.mesh, 1/Lmean)
    return I

def init_corners_and_features(I : Instance, only_border: bool, verbose: bool):
    """Initializes features and cut the mesh to create the work mesh"""

    I.feat = FeatureCutter(I.mesh, only_border=only_border, corner_order=I.order, verbose=verbose)()
    I.work_mesh = I.feat.cut_mesh # we split the mesh along feature edges

    vertex_cut_attr = I.work_mesh.vertices.create_attribute("cuts", int)
    for i,(v,split_v) in enumerate(I.feat.cut_vertices.items()):
        for b in split_v:
            vertex_cut_attr[b] = i+1

def init_attributes(I : Instance):
    """Initializes attributes of the work mesh (input mesh cut along features)"""
    I.barycenters = M.attributes.face_barycenter(I.work_mesh, persistent=False)
    I.areas = M.attributes.face_area(I.work_mesh, persistent=False)
    
    M.attributes.face_normals(I.work_mesh) # initialize face normals (used several times afterwards by other attributes)
    I.angles = M.attributes.corner_angles(I.work_mesh, persistent=False)
    I.vnormals = M.attributes.vertex_normals(I.work_mesh)
    
    # build defects from angles: 
    I.defect = I.work_mesh.vertices.create_attribute("defect", float, dense=True)
    for C,V in enumerate(I.work_mesh.face_corners):
        I.defect[V] += I.angles[C]

def init_local_bases(I : Instance, free_bnd):
    """ Initializes natural parallel transport """
    I.connection = M.processing.SurfaceConnectionVertices(I.work_mesh, feat=I.feat, vnormal=I.vnormals, angles=I.angles)

    if free_bnd:
        for u in I.mesh.id_vertices:
            # initialize angles of every edge in this basis -> flatten to 2pi
            ang = 0.
            bnd = I.mesh.is_vertex_on_border(u)
            for v in I.mesh.connectivity.vertex_to_vertex(u):
                T = I.mesh.half_edges.adj(u,v)[0]
                if bnd:
                    I.connection._transport[(u,v)] = ang * pi / I.defect[u]
                else:
                    I.connection._transport[(u,v)] = ang * 2 * pi / I.defect[u]
                c = I.mesh.connectivity.vertex_to_corner_in_face(u,T)
                if c is None : continue
                ang += I.angles[c]

    # compute parallel transport array
    I.PT_array = np.zeros(len(I.work_mesh.edges), dtype=np.float64)
    for e,(A,B) in enumerate(I.work_mesh.edges):
        I.PT_array[e] = principal_angle(I.connection.transport(B,A) - I.connection.transport(A,B))

    # compute curvature on the mesh
    if I.work_mesh.faces.has_attribute("curvature"):
        I.curvature = I.work_mesh.faces.get_attribute("curvature")
    else:
        I.curvature = I.work_mesh.faces.create_attribute("curvature", float)
        for iF,F in enumerate(I.work_mesh.faces):
            v = 1+0j
            for a,b in M.utils.cyclic_pairs(F):
                v *= cmath.rect(1., I.connection.transport(b,a)- I.connection.transport(a,b) - pi)
            I.curvature[iF] = cmath.phase(v)

def init_variables_edges(I : Instance, free_bnd):
    """Initializes variables of the parametrization as well as indirections to retrieve edges or charts"""
    I.var_sep_pt = 4*len(I.work_mesh.edges)
    n_edge_var = 4*len(I.work_mesh.edges) + 6*len(I.work_mesh.faces)
    I.nvar += n_edge_var
    var_edge = np.zeros(n_edge_var, dtype=np.float64)
    EL = M.attributes.edge_length(I.work_mesh, persistent=False) # Edge Lengths
    I.edge_lengths = -1*np.ones((len(I.work_mesh.edges), 3), dtype=np.float64)

    for u, Pu in enumerate(I.work_mesh.vertices):
        edges_u = I.work_mesh.connectivity.vertex_to_edge(u) # the edges around u
        for ie in edges_u:
            e = I.work_mesh.edges[ie]
            direct = (e[0]==u) # edge is (u,v) or (v,u)
            v = e[1] if direct else e[0] # other extremity of edge
            Pv = I.work_mesh.vertices[v]
            elen = EL[ie]/2
            I.edge_lengths[ie,0] = elen
            eangle = I.connection.transport(u,v)

            evar = cmath.rect(elen, eangle)
            if direct:
                var_edge[4*ie]   = evar.real
                var_edge[4*ie+1] = evar.imag
            else:
                var_edge[4*ie+2] = evar.real
                var_edge[4*ie+3] = evar.imag
            
            for iw,w in enumerate((I.work_mesh.half_edges.next_around(u,v), I.work_mesh.half_edges.prev_around(u,v))):
                if w is not None:
                    T = I.work_mesh.connectivity.face_id(u,v,w)
                    iuT = I.work_mesh.connectivity.in_face_index(T,u)
                    Pbary = I.barycenters[T]
                    slen = geom.distance(Pbary, Pu)
                    sangle = geom.angle_3pts(Pv, Pu, Pbary)
                    if free_bnd:
                        if I.mesh.is_vertex_on_border(u):
                            sangle *= pi/ I.defect[u]
                        else:
                            sangle *= 2*pi / I.defect[u]
                    else:
                        if u in I.feat.feature_vertices:
                            sangle *= (pi/I.order * I.feat.corners[u]) / I.defect[u]
                        else:
                            sangle *= 2*pi / I.defect[u]
                    svar = cmath.rect(slen, eangle+sangle) if iw==0 else cmath.rect(slen, eangle-sangle)
                    var_edge[I.var_sep_pt+6*T+2*iuT] = svar.real
                    var_edge[I.var_sep_pt+6*T+2*iuT+1] = svar.imag
                    I.edge_lengths[ie,iw+1] = abs(svar - evar)
    return var_edge

def init_var_ff_on_feat(I : Instance, var_edge):
    """
    Initializes frame field variables. Zero everywhere except on boundary and features edges where the frame follows the edge.
    """
    I.var_sep_ff = I.nvar
    ffvar = np.zeros(len(I.work_mesh.vertices), dtype=complex)
    for v in I.feat.feature_vertices:
        for ie in I.feat.local_feat_edges[v]:
            e = I.work_mesh.connectivity.vertex_to_edge(v)[ie]
            direct = I.work_mesh.edges[e][0]==v
            nmid = 4*e if direct else 4*e + 2
            zv = complex(var_edge[nmid], var_edge[nmid + 1])
            ffvar[v] += (zv/abs(zv))**I.order
    for v in I.feat.feature_vertices:
        if abs(ffvar[v])>1e-8:
            ffvar[v] /= abs(ffvar[v])
        else:
            ffvar[v] = complex(1., 0.)
    # for v in self.feat.feature_vertices:
    #     ffvar[v] = 1+0j

    if len(I.feat.feature_vertices)==0:
        # we are borderless with no feature edges -> we need an arbitrary fixed point for the frame field
        I.feat.feature_vertices = {0}
        I.feat.local_feat_edges[0] = []
        ffvar[0] = complex(1.,0.)

    var_ff = np.zeros(2*ffvar.size, dtype=np.float64)
    for c in range(len(I.work_mesh.vertices)):
        var_ff[2*c] = ffvar[c].real
        var_ff[2*c+1] = ffvar[c].imag
    I.nvar += 2*len(I.work_mesh.vertices)
    return var_ff

def init_var_rotations(I : Instance, var_ff):
    """
    Initializes rotations variables.
    - Fixed on boundary and feature edges to match the frame field.
    - other values are determined by a least square problem to alter the parallel transport and push singularities on the boundary
    """
    I.var_sep_rot = I.nvar # number of variables so far
    n = len(I.work_mesh.edges)
    var_rot = np.zeros(n, dtype=np.float64)
    I.nvar += n
    
    ## build angles on feature edges
    for e in I.feat.feature_edges:
        A,B = I.work_mesh.edges[e]
        aA,aB = I.connection.transport(A,B), I.connection.transport(B,A) # local basis orientation for A and B
        fA = complex(var_ff[2*A], var_ff[2*A+1]) # representation complex for frame field at A
        fB = complex(var_ff[2*B], var_ff[2*B+1]) # representation complex for frame field at B
        uB = roots(fB, I.order)[0]
        abs_angles = [abs(angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA)) for uA in roots(fA, I.order)]
        angles = [angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA) for uA in roots(fA, I.order)]
        i_angle = np.argmin(abs_angles)
        w = -angles[i_angle]
        var_rot[e] = w

    ## build start w for rotation penalty energy
    # constraints
    cstrfaces = M.attributes.faces_near_border(I.work_mesh, 4)

    ncstr = len(I.feat.feature_edges)
    nvar = len(I.work_mesh.edges)
    Cstr = sp.lil_matrix((ncstr, nvar))
    Rhs = np.zeros(ncstr)
    for i,e in enumerate(I.feat.feature_edges):
        Cstr[i,e] = 1
        Rhs[i] = var_rot[e]

    # objective
    D1 = sp.lil_matrix((len(I.work_mesh.faces), nvar))
    b1 = np.zeros(len(I.work_mesh.faces))
    for iT,T in enumerate(I.work_mesh.faces):
        for (u,v) in M.utils.cyclic_pairs(T):
            e = I.work_mesh.connectivity.edge_id(u,v)
            D1[iT, e] = 1 if u<v else -1
        b1[iT] = -I.curvature[iT]
    
    D2 = sp.lil_matrix((len(cstrfaces), nvar))
    b2 = np.zeros(len(cstrfaces))
    for i,f in enumerate(cstrfaces):
        A,B,C = I.work_mesh.faces[f]
        for u,v in [(A,B), (B,C), (C,A)]:
            e = I.work_mesh.connectivity.edge_id(u,v)
            D2[i,e] = 1e3 if (u<v) else -1e3
        b2[i] = -1e3*I.curvature[f]

    Id = sp.identity(len(I.work_mesh.edges), format="csc")
    b3 = np.zeros(len(I.work_mesh.edges))

    P = sp.vstack((D1,D2,Id))
    b = np.concatenate([b1,b2,b3])
    b = P.transpose().dot(b)
    P = P.transpose().dot(P)
    osqp_instance = OSQP()
    osqp_instance.setup(P, b, A=Cstr.tocsc(), l=Rhs, u=Rhs, verbose=False)
    res = osqp_instance.solve().x
    
    for e in I.work_mesh.id_edges:
        var_rot[e] = math.tan(res[e]/2) if res is not None else 0.
    return var_rot

def init_var_ff_and_rot_full(I : Instance, compute_singus : bool, verbose : bool = False):
    """
    Computes a frame field on the mesh and initializes variables of the frame field and rotations according to this fixed frame field.
    """

    ff = M.framefield.SurfaceFrameField(
        I.work_mesh, 
        "vertices", 
        I.order, 
        features=True,
        cad_correction=True,
        smooth_normals=False,
        custom_connection=I.connection,
        custom_features=I.feat,
        verbose=verbose
    )

    ff.initialize()
    ff.optimize()
    ff.flag_singularities() # build attributes 'faces.singuls' and 'edges.angles'
    if compute_singus:
        I._singular_faces = I.work_mesh.faces.get_attribute("singuls") # retrieve singularities from the attribute created by flag_singularities
    print(f"Found {len([x for x in I.work_mesh.faces.get_attribute('singuls')])} singularity cones")
    angles = I.work_mesh.edges.get_attribute("angles") # retrieve angles from the attribute created by flag_singularities

    # fill var arrays
    I.var_sep_ff = I.nvar
    var_ff = np.zeros(2*len(I.work_mesh.vertices), dtype=np.float64)
    I.nvar += 2*len(I.work_mesh.vertices)
    for v in I.work_mesh.id_vertices:
        var_ff[2*v] = ff[v].real
        var_ff[2*v+1] = ff[v].imag

    I.var_sep_rot = I.nvar
    var_rot = np.zeros(len(I.work_mesh.edges), dtype=np.float64)
    I.nvar += len(I.work_mesh.edges)
    for e in I.work_mesh.id_edges:
        ang = angles[e]
        var_rot[e] = math.tan(ang/2)

    if len(I.feat.feature_vertices)==0:
        # we are borderless with no feature edges -> we need an arbitrary fixed point for the frame field
        for v in I.work_mesh.id_vertices:
            if abs(ff[v])>1e-8:
                I.feat.feature_vertices = {v}
                I.feat.local_feat_edges[v] = []
                break
    return var_ff, var_rot

def init_var_ff_and_rot_curvature(I : Instance, optim_fixed_ff:bool):
    assert I.order == 4

    ff = M.framefield.PrincipalDirections(I.work_mesh,"vertices", features=True, curv_threshold=0.08, patch_size=5, n_smooth=0, custom_features=I.feat) 
    ff.initialize()
    ff.optimize()
    ff.flag_singularities() # build attributes 'faces.singuls' and 'edges.angles'
    if optim_fixed_ff:
        I._singular_faces = I.work_mesh.faces.get_attribute("singuls") # retrieve singularities from the attribute created by flag_singularities
    print(f"Found {len([x for x in I.work_mesh.faces.get_attribute('singuls')])} singularity cones")
    angles = I.work_mesh.edges.get_attribute("angles") # retrieve angles from the attribute created by flag_singularities

    # fill var arrays
    I.var_sep_ff = I.nvar
    I.nvar += 2*len(I.work_mesh.vertices)
    var_ff = np.zeros(2*len(I.work_mesh.vertices), dtype=np.float64)
    for v in I.work_mesh.id_vertices:
        var_ff[2*v] = ff[v].real
        var_ff[2*v+1] = ff[v].imag
    
    I.var_sep_rot = I.nvar
    I.nvar += len(I.work_mesh.edges)
    var_rot = np.zeros(len(I.work_mesh.edges), dtype=np.float64)
    for e in I.work_mesh.id_edges:
        ang = angles[e]
        var_rot[e] = math.tan(ang/2)
    
    if len(I.feat.feature_vertices)==0:
        # we are borderless with no feature edges -> we need an arbitrary fixed point for the frame field
        I.feat.feature_vertices = {0}
        I.feat.local_feat_edges[0] = []
    return var_ff, var_rot

def init_var_ff_and_rot_random(I : Instance, var_edge):
    I.var_sep_ff = I.nvar
    I.nvar += 2*len(I.work_mesh.vertices)
    var_ff = np.zeros(2*len(I.work_mesh.vertices), dtype=np.float64)

    I.var_sep_rot = I.nvar
    I.nvar += len(I.work_mesh.edges)
    var_rot = np.zeros(len(I.work_mesh.edges), dtype=np.float64)

    ff = (2*pi/ I.order) * np.random.random(len(I.work_mesh.vertices))
    ff = np.array([ cmath.rect(1, I.order*_a) for _a in ff])
    for v in I.work_mesh.id_vertices:
        var_ff[2*v] = ff[v].real
        var_ff[2*v+1] = ff[v].imag
    for v in I.feat.feature_vertices:
        var_ff[2*v] = 0
        var_ff[2*v+1] = 0
        for ie in I.feat.local_feat_edges[v]:
            e = I.work_mesh.connectivity.vertex_to_edge(v)[ie]
            direct = I.work_mesh.edges[e][0]==v
            nmid = 4*e if direct else 4*e + 2
            zv = complex(var_edge[nmid], var_edge[nmid+1])
            c = (zv/abs(zv))**4
            var_ff[2*v] += c.real
            var_ff[2*v+1] += c.imag
    for v in I.feat.feature_vertices:
        nv = var_ff[2*v]**2 + var_ff[2*v+1]**2
        if abs(nv)>1e-8:
            var_ff[2*v] /= np.sqrt(nv)
            var_ff[2*v+1] /= np.sqrt(nv)
        else:
            var_ff[2*v] = 1.
            var_ff[2*v+1] = 0.

    # compute the rotation induced by the frame field on every edge
    for e,(A,B) in enumerate(I.work_mesh.edges):
        fA = complex(var_ff[2*A], var_ff[2*A+1])
        fB = complex(var_ff[2*B], var_ff[2*B+1]) # representation complex for A and B
        aA,aB = I.connection.transport(A,B), I.connection.transport(B,A) # local basis orientation for A and B
        uB = roots(fB, 4)[0]
        abs_angles = [abs( angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA)) for uA in roots(fA, 4)]
        angles = [angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA) for uA in roots(fA, 4)]
        i_angle = np.argmin(abs_angles)
        var_rot[e] = -math.tan(angles[i_angle]/2)
    
    if len(I.feat.feature_vertices)==0:
        # we are borderless with no feature edges -> we need an arbitrary fixed point for the frame field
        I.feat.feature_vertices = {0}
        I.feat.local_feat_edges[0] = []
    return var_ff, var_rot

def init_var_ff_and_rot_free_bnd(I : Instance):
    """
    Frames not aligned with boundary + no cones
    Simples implementation of Trivial Connection on Surfaces by Crane et al.
    """
    n = len(I.mesh.vertices)
    I.var_sep_ff = I.nvar
    I.nvar += 2*n
    var_ff = np.zeros(2*n, dtype=np.float64)

    # Initialize rotations
    # min ||w||^2 s.t. dw = curvature
    I.var_sep_rot = I.nvar
    m = len(I.mesh.edges)
    I.nvar += m

    Cst = sp.lil_matrix((len(I.mesh.faces), m))
    K = np.zeros(len(I.mesh.faces))
    for iF in I.mesh.id_faces:
        A,B,C = I.mesh.faces[iF]
        for u,v in [(A,B), (B,C), (C,A)]:
            e = I.mesh.connectivity.edge_id(u,v)
            Cst[iF,e] = 1 if (u<v) else -1
        K[iF] = I.curvature[iF]

    osqp_instance = OSQP()
    osqp_instance.setup(sp.eye(m,format="csc"), A=Cst.tocsc(), l=K, u=K, verbose=False)
    omegas = osqp_instance.solve().x
    var_rot = np.zeros_like(omegas)
    for e in I.mesh.id_edges:
        var_rot[e] = math.tan(omegas[e]/2)

    # Initialize frame field using a tree
    tree = M.processing.trees.EdgeSpanningTree(I.mesh, 0, True)()
    for j,i in tree.traverse():
        if i is None : # root
            var_ff[2*j], var_ff[2*j+1] = 1., 0.
            continue
        e = I.mesh.connectivity.edge_id(i,j)
        aij, aji = I.connection.transport(i,j), I.connection.transport(j,i)
        ci = complex(var_ff[2*i], var_ff[2*i+1])
        cj = ci * cmath.rect(1, omegas[e] + aji - aij - pi) if j<i else ci * cmath.rect(1, -omegas[e] + aji - aij - pi)
        var_ff[2*j] = cj.real
        var_ff[2*j+1] = cj.imag
    return var_ff, var_rot

def init_distortion_matrices(I : Instance):
    mats = []
    for (iS, ie1, ie2) in I.quads:
        # uS,vS = I.init_var[iS], I.init_var[iS+1]
        ue1, ve1 = I.init_var[ie1], I.init_var[ie1+1]
        ue2, ve2 = I.init_var[ie2], I.init_var[ie2+1]

        m1 = np.array([[ue1, ue2], 
                       [ve1, ve2]])
        m1_inv = np.linalg.inv(m1)
        mats.append(m1_inv)
    I.dist_matrices = np.array(mats)

def init_edge_indices(I : Instance):
    N = len(I.work_mesh.edges)
    M = len(I.work_mesh.faces)
    I.edge_indices = np.zeros((3*M, 3), dtype=np.int32)
    new_lengths = np.zeros(N+3*M, dtype=np.float64)
    len_i = 0
    for ie,(A,B) in enumerate(I.work_mesh.edges):
        new_lengths[len_i] = I.edge_lengths[ie,0]
        len_i += 1

    # then all the pi/pj or qi/qj pairs
    ind_i = 0
    for T in I.work_mesh.id_faces:
        for ie in I.work_mesh.connectivity.face_to_edge(T):
            A,B = I.work_mesh.edges[ie]
            iA = I.work_mesh.connectivity.in_face_index(T,A)
            iB = I.work_mesh.connectivity.in_face_index(T,B)
            I.edge_indices[ind_i, 0] = ie
            I.edge_indices[ind_i, 1] = I.var_sep_pt + 6*T  + 2*iA
            I.edge_indices[ind_i, 2] = I.var_sep_pt + 6*T  + 2*iB
            if I.work_mesh.half_edges.adj(A,B)[0]==T:
                new_lengths[len_i] = I.edge_lengths[ie,1]
            else:
                new_lengths[len_i] = I.edge_lengths[ie,2]
            ind_i += 1
            len_i += 1
    #I.edge_lengths = np.ones_like(new_lengths, dtype=np.float64)
    I.edge_lengths = new_lengths

def initialize_quad_indices_and_ref_dets(I : Instance, var):
    N = 3*len(I.work_mesh.faces)
    I.quad_indices = np.zeros((N,3), dtype=np.int32)
    I.ref_dets = np.zeros((N,3), dtype=np.float64)
    k = 0
    for (iS, ie1, ie2) in I.quads:
        I.quad_indices[k,:] = [iS, ie1, ie2]

        pS  = M.Vec(var[iS], var[iS+1])
        pe1 = M.Vec(var[ie1], var[ie1+1])
        pe2 = M.Vec(var[ie2], var[ie2+1])
        det0 = geom.det_2x2(pe1, pe2)
        det1 = geom.det_2x2(pe1, pS)
        det2 = geom.det_2x2(pe2, pS)
        I.ref_dets[k,:] = [det0, det1, det2]
        k += 1

def initialize_rotFF_indices(I : Instance):
    N = len(I.work_mesh.edges)
    I.rotFF_indices = np.zeros((N,3), dtype=np.int32)
    for e,(A,B) in enumerate(I.work_mesh.edges):
        I.rotFF_indices[e,0] = I.var_sep_rot + e
        I.rotFF_indices[e,1] = I.var_sep_ff + 2*A
        I.rotFF_indices[e,2] = I.var_sep_ff + 2*B 

##### Worker #####

class Initializer(Worker):
    """A Worker that is responsible for the initialization of an instance, including:
        - Various attributes of the mesh (normals, boundaries, barycenters,...)
        - All variables (points, rotations and frame field)
    """

    def __init__(self, instance: Instance, options = Options(), verbose_options = VerboseOptions()):
        super().__init__("Initializer", instance, options, verbose_options)

    def __call__(self):
        self.initialize()
        return self.instance

    def initialize(self):
        self.log("Initializing")

        self.log("Initialize attributes of original mesh")
        rescale_input_mesh(self.instance)

        self.log("Cut corners and initialize feature edges")
        init_corners_and_features(self.instance, not self.options.features, self.verbose_options.logger_verbose) # /!\ changes the combinatorics of the mesh

        self.log("Initialize attributes of working mesh")
        init_attributes(self.instance)

        self.log("Initialize local flattened bases")
        init_local_bases(self.instance, self.options.free_boundary)

        self.log("Initialize variables")
        # /!\ respect order of initialization
        var_edges = init_variables_edges(self.instance, self.options.free_boundary)
        
        if self.options.free_boundary:
            var_ff,var_rot = init_var_ff_and_rot_free_bnd(self.instance)
        else:
            if self.options.initMode == InitMode.AUTO:
                if self.options.features or len(self.instance.mesh.boundary_vertices)>0:
                    # init zero
                    var_ff  = init_var_ff_on_feat(self.instance, var_edges) 
                    var_rot = init_var_rotations(self.instance, var_ff)
                else:
                    # init smooth
                    var_ff, var_rot = init_var_ff_and_rot_full(self.instance, self.options.optimFixedFF, self.verbose_options.logger_verbose)
            elif self.options.initMode == InitMode.CURVATURE:
                var_ff, var_rot = init_var_ff_and_rot_curvature(self.instance, self.options.optimFixedFF)
            elif self.options.initMode == InitMode.SMOOTH:
                var_ff, var_rot = init_var_ff_and_rot_full(self.instance, self.options.optimFixedFF, self.verbose_options.logger_verbose)
            elif self.options.initMode == InitMode.RANDOM:
                var_ff, var_rot = init_var_ff_and_rot_random(self.instance, var_edges)
            else: # init mode is zero
                var_ff  = init_var_ff_on_feat(self.instance, var_edges) 
                var_rot = init_var_rotations(self.instance, var_ff)
        
        self.instance.var = np.concatenate((var_edges, var_ff, var_rot))
        self.instance.init_var = np.copy(self.instance.var)

        initialize_rotFF_indices(self.instance)
        init_edge_indices(self.instance)
        initialize_quad_indices_and_ref_dets(self.instance, var_edges)
        
        if self.options.distortion == Distortion.ARAP:
            self.log("Compute initial jacobians for distortion")
            init_distortion_matrices(self.instance)

        self.log(f"Number of variables: {self.instance.var.size}")

        self.instance.initialized = True
        self.log("Initialisation done\n")