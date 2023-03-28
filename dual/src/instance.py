import mouette as M
from mouette import geometry as geom

import math
import cmath
import numpy as np

from .featurecut import FeatureCutter

class Instance:
    """
    Instance of our problem.
    Contains every data needed for optimization and reconstruction
    """
    def __init__(self, mesh : M.mesh.SurfaceMesh):
        self.mesh : M.mesh.SurfaceMesh = mesh # the original mesh
        self.work_mesh : M.mesh.SurfaceMesh = None # the working mesh (combinatorics change due to feature edges)
        self.param_mesh : M.mesh.SurfaceMesh = None # the parametrization as if (u,v) = (x,y). Debug output
        self.disk_mesh : M.mesh.SurfaceMesh = None # the mesh with uvs but with disk topology (cut along all seams). Debug output
        self.cut_graph : M.mesh.PolyLine = None # seams
        self.singu_ptcld : M.mesh.PointCloud = None # singularities as point cloud

        self.feat : FeatureCutter = None

        self.order = 4 # order of the frame field

        # attributes (/!\ on work_mesh)
        self.curvature : M.Attribute = None # gaussian curvature at vertices
        self.barycenters : M.Attribute = None # barycenters of triangles
        self.areas : M.Attribute = None # areas of triangles
        self.edge_lengths : np.ndarray = None # length of all edges
        self.angles : M.Attribute = None # angle at each triangle corners
        self.defect : M.Attribute = None # sum of angles around a vertex (/!\ not 'real' defect which is 2*pi - this)
        self.parallel_transport = dict()
        self.PT_array : np.ndarray = None # parallel transport as a numpy array
        self.PT_array2 : np.ndarray = None # the two are different if we initialize following curvature
        self.cotan_weight : np.ndarray = None

        self._singular_faces : M.Attribute = None # boolean flag on faces. Accessed via a property
        self.singular_vertices : M.Attribute = None # boolean flag on vertices

        # soft feature alignment
        self.soft_feat_indices : np.ndarray = None
        self.soft_feat_indices_ff : np.ndarray = None
        self.soft_feat_normals : np.ndarray = None
        self.soft_feat_ff : np.ndarray = None

        # Local bases
        self.vbaseX : M.Attribute = None # local basis X vector (tangent)
        self.vbaseY : M.Attribute = None # local basis Y vector (tangent)
        self.vnormals : M.Attribute = None # local basis Z vector (normal)

        # Distortion and barriers
        self.init_var : np.ndarray = None # copy of self.var before optimizing. Kept as a reference.
        self.ref_dets : np.ndarray = None
        self.dist_matrices : np.ndarray = None

        # curvature constraints
        self.curvFF : M.processing.framefield.CurvatureVertices = None
        self.curvFFvar : np.ndarray = None

        # reconstruction
        self.tree : M.processing.trees.FaceSpanningTree = None
        self.singu_faces_to_4pts : dict = None
        self.triplets_of_triangles : dict = None
        self.triangles_before_split : dict = None
        self.UVs : M.Attribute = None
        self.seams : set = None

        # variables
        self.var : np.ndarray = None
        self.nvar : int = 0
        self.var_sep_pt = 4*len(self.mesh.edges)
        self.var_sep_rot : int = 0 # separator between mesh vars and rotations (first index of a rotation)
        self.var_sep_ff : int = 0 # separator between rotations vars and framefield var (first index of a ff var)
        self.var_sep_scale : int = 0 # separator between framefield var and scale var (first index of a scale)

        self.edge_indices : np.ndarray = None # indices necessary for edge constraint energy
        self.rotFF_indices : np.ndarray = None # indices necessary for rot_follow_FF energy
        self.quad_indices : np.ndarray = None
        self.scale_indices : np.ndarray = None

        self.initialized = False # init flag

    def get_var_rot(self,e):
        return self.var[self.var_sep_rot + e]

    def get_rotation(self, e):
        # variables are tan(a/2)
        return 2*math.atan(self.get_var_rot(e))

    def get_var_ff(self,v):
        return self.var[self.var_sep_ff + 2*v: self.var_sep_ff + 2*(v+1)]
            
    @property
    def quads(self):
        for T in self.work_mesh.id_faces:
            for A,B,C in M.utils.cyclic_permutations(self.work_mesh.faces[T]):
                # B is central vertex
                eBA = self.work_mesh.connectivity.edge_id(A,B)
                eBC = self.work_mesh.connectivity.edge_id(C,B)
                iS = self.var_sep_pt + 6*T+2*self.work_mesh.connectivity.in_face_index(T,B)
                ie1 = 4*eBA if B<A else 4*eBA + 2
                ie2 = 4*eBC if B<C else 4*eBC + 2
                yield iS, ie1, ie2

    @property
    def singular_faces(self):
        if self._singular_faces is None:
            # compute
            self._singular_faces = self.mesh.faces.create_attribute("singularities", int)
            
            ZERO_THRESHOLD = 1e-3
            if self.work_mesh.faces.has_attribute("defect"):
                defect = self.work_mesh.faces.get_attribute("defect")
            else:
                defect = self.work_mesh.faces.create_attribute("defect", float)
            for iF,(A,B,C) in enumerate(self.work_mesh.faces):
                angle = 0
                for u,v in [(A,B), (B,C), (C,A)]:
                    e = self.work_mesh.connectivity.edge_id(u,v)
                    w = self.get_rotation(e)
                    angle += w if u>v else -w
                angle += self.curvature[iF] # /!\ very important to counteract effect of natural curvature
                defect[iF] = angle
                if abs(angle)>ZERO_THRESHOLD:
                    self._singular_faces[iF] = round(angle / (2 * math.pi / self.order))
        return self._singular_faces

    def export_frame_field(self):
        """
        Exports the frame field as a mesh for visualization.

        Returns:
            PolyLine: the frame field as a mesh object
        """
        FFMesh = M.mesh.new_polyline()
        L = M.attributes.mean_edge_length(self.work_mesh)/4
        for id_vertex, P in enumerate(self.work_mesh.vertices):
            if id_vertex >= len(self.vbaseX): continue
            E,N = self.vbaseX[id_vertex], self.vnormals[id_vertex]
            if geom.norm(N)<1e-8: continue
            z = complex(self.var[self.var_sep_ff + 2*id_vertex], self.var[self.var_sep_ff + 2*id_vertex + 1])
            angle = cmath.phase(z)/self.order
            cmplx = [geom.rotate_around_axis(E, N, angle + k * 2 * math.pi / self.order) for k in range(self.order)]
            pts = [P + abs(z)*L*r for r in cmplx]
            FFMesh.vertices.append(P)
            FFMesh.vertices += pts
            n = self.order+1
            FFMesh.edges += [(n*id_vertex, n*id_vertex+k) for k in range(1,n)]
        return FFMesh