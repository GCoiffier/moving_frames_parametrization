import mouette as M
from mouette import geometry as geom
from .worker import *
from .instance import Instance
from math import atan2, pi, tan
import numpy as np

def rescale_input_mesh(I : Instance):
    """Rescales the mesh to avoid numerical problems"""
    Lmean = M.attributes.mean_edge_length(I.mesh)
    I.mesh = M.transform.scale(I.mesh, 1/Lmean)
    return I

def initialize_features(I : Instance, features:bool, verbose:bool):
    I.feat = M.processing.FeatureEdgeDetector(only_border = not features, verbose=verbose)(I.mesh)
    split = False
    # subdivide triangles with two or more feature edges
    with M.processing.SurfaceSubdivision(I.mesh) as subdiv:
        for iF in I.mesh.id_faces:
            eF = I.mesh.connectivity.face_to_edge(iF)
            n_adj_feat = sum([e in I.feat.feature_edges for e in eF])
            if n_adj_feat>1:
                subdiv.split_face_as_fan(iF)
                split = True
    if split:
        I.mesh.connectivity.clear()
        I.mesh.half_edges.clear()
        I.mesh.clear_boundary_data()
        I.feat.run(I.mesh) # should not be necessary
    return I

def initialize_attributes(I : Instance):
    I.defect = M.attributes.angle_defects(I.mesh,persistent=False)
    I.connection = M.processing.SurfaceConnectionFaces(I.mesh, I.feat)
    I.PT_array = np.zeros(len(I.mesh.interior_edges), dtype=np.float64)
    for i,ie in enumerate(I.mesh.interior_edges):
        ea,eb = I.mesh.edges[ie]
        T1,T2 = I.mesh.half_edges.edge_to_triangles(ea,eb)
        I.PT_array[i] = I.connection.transport(T2,T1)
    return I

def initialize_edge_indices(I : Instance):
    N = len(I.mesh.interior_edges)
    I.ref_edges = np.zeros((N,2,2), dtype=np.float64)
    I.edge_indices = np.zeros((N,3), dtype=np.int32)

    for i,e in enumerate(I.mesh.interior_edges):
        v1,v2 = I.mesh.edges[e]
        T1,T2 = I.mesh.half_edges.edge_to_triangles(v1,v2)
        P1,P2 = (I.mesh.vertices[_v] for _v in (v1,v2))
        E = M.Vec(P2-P1)
        assert T1 is not None and T2 is not None

        X1,Y1 = I.local_base(T1)
        ET1 = M.Vec(X1.dot(E), Y1.dot(E))
        ET1.normalize()
        X2,Y2 = I.local_base(T2)
        ET2 = M.Vec(X2.dot(E), Y2.dot(E))
        ET2.normalize()

        I.ref_edges[i,0,:] = ET1
        I.ref_edges[i,1,:] = ET2
        I.edge_indices[i] = [e, T1, T2]
    return I

def initialize_var_jacobian(I : Instance):
    var = np.zeros(4*len(I.mesh.faces))
    for iT in I.mesh.id_faces:
        # a c  =  1 0
        # b d     0 1
        # identity jacobian in column order
        var[4*iT] = 1.
        var[4*iT+3] = 1.
    I.nvar += var.size
    return var

def initialize_var_ff_on_feat(I : Instance):
    I.var_sep_ff = I.nvar
    var = np.zeros(2*len(I.mesh.faces))
    I.nvar += var.size

    # fix orientation on features
    for e in I.feat.feature_edges:
        e1,e2 = I.mesh.edges[e] # the edge on border
        edge = I.mesh.vertices[e2] - I.mesh.vertices[e1]
        for T in I.mesh.half_edges.edge_to_triangles(e1,e2):
            if T is None: continue # edge may be on boundary
            X,Y = I.local_base(T)
            c = complex(edge.dot(X), edge.dot(Y)) # compute edge in local basis coordinates (edge.dot(Z) = 0 -> complex number for 2D vector)
            c = (c/abs(c))**4 # c^4 is the same for all four directions of the cross
            var[2*T] = c.real
            var[2*T+1] = c.imag 
    
    # if no features and no boundary -> fix a random frame
    if len(I.feat.feature_vertices)==0:
        # we choose arbitrarily frame 0
        var[0] = 1. # fix value at 1+0i
    return var

def initialize_var_rotations(I : Instance):
    I.var_sep_rot = I.nvar
    var = np.zeros(len(I.mesh.edges))
    I.nvar += var.size
    return var

def initialize_var_ff_trivial_connection(I: Instance):
    I.var_sep_ff = I.nvar
    I.nvar += 2*len(I.mesh.faces)
    I.var_sep_rot = I.nvar
    I.nvar += len(I.mesh.edges)
    
    singus = I.mesh.vertices.create_attribute("singu_cstr", int)
    for s in I.mesh.vertices.get_attribute("selection"):
        singus[s] = 1
    singus[282] = -4
    ff = M.framefield.TrivialConnectionFaces(I.mesh, singus_indices=singus)()
    
    var_ff = np.zeros(2*len(I.mesh.faces))
    for f in I.mesh.id_faces:
        var_ff[2*f] = ff[f].real
        var_ff[2*f+1] = ff[f].imag
    var_rot = -np.tan(ff.rotations/2)
    return var_ff, var_rot


def initialize_var_ff_fixed(I: Instance, feat:bool, verbose:bool, compute_singus):
    ff = M.framefield.SurfaceFrameField(I.mesh, "faces", features=feat, verbose=verbose, custom_connection=I.connection, custom_feature=I.feat)()
    ff.flag_singularities()
    
    if compute_singus:
        I._singular_vertices = I.mesh.vertices.get_attribute("singuls")

    I.var_sep_ff = I.nvar
    var_ff = np.zeros(2*len(I.mesh.faces))
    for iT in I.mesh.id_faces:
        var_ff[2*iT] = ff.var[iT].real
        var_ff[2*iT+1] = ff.var[iT].imag

    I.nvar += var_ff.size
    I.var_sep_rot = I.nvar
    var_rot = np.zeros(len(I.mesh.edges))
    edge_rot = I.mesh.edges.get_attribute("angles")
    for ie in I.mesh.id_edges:
        var_rot[ie] = -tan(edge_rot[ie]/2)
    I.nvar += var_rot.size
    return var_ff, var_rot

def initialize_ff_indices(I : Instance):
    I.ff_indices = np.zeros((len(I.mesh.interior_edges), 3), dtype=np.int32)
    for i,e in enumerate(I.mesh.interior_edges):
        I.ff_indices[i,0] = e
        A,B = I.mesh.edges[e]
        T1,T2 = I.mesh.half_edges.edge_to_triangles(A,B)
        I.ff_indices[i,1] = I.var_sep_ff + 2*T1
        I.ff_indices[i,2] = I.var_sep_ff + 2*T2
    return I

class Initializer(Worker):

    def __init__(self, instance: Instance, options : Options, verbose_options : VerboseOptions):
        super().__init__("Initializer", instance, options, verbose_options)

    def __call__(self):
        self.initialize()
        return self

    def initialize(self):
        verb = self.verbose_options.logger_verbose
        self.log("Initializing")
        rescale_input_mesh(self.instance)

        self.log("Initialize various attributes on the mesh")
        initialize_features(self.instance, features = self.options.features, verbose=verb)
        initialize_attributes(self.instance)
        
        self.log("Initialize variables")
        initialize_edge_indices(self.instance)
        var_jac = initialize_var_jacobian(self.instance) # same init whatever the init mode
        if self.options.initSmooth:
            var_ff, var_rot = initialize_var_ff_fixed(self.instance, self.options.features, verb, self.options.optimFixedFF) # inits both ff and rotations
        else:
            var_ff = initialize_var_ff_on_feat(self.instance)
            var_rot = initialize_var_rotations(self.instance)
        self.instance.var = np.concatenate([var_jac, var_ff, var_rot]).astype(np.float64)

        initialize_ff_indices(self.instance)
        self.instance.initialized = True
        self.log("Instance Initialized")