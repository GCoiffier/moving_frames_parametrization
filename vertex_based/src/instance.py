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
        self.work_mesh : M.mesh.SurfaceMesh = None # the working mesh (combinatorics change due to feature edges being split and counted as boundary)
        self.param_mesh : M.mesh.SurfaceMesh = None # the parametrization as if (u,v) = (x,y). Debug output
        self.disk_mesh : M.mesh.SurfaceMesh = None # the mesh with uvs but with disk topology (cut along all seams). Debug output
        self.cut_graph : M.mesh.PolyLine = None # seams as line graph
        self.singu_ptcld : M.mesh.PointCloud = None # singularities as point cloud

        self.feat : FeatureCutter = None

        self.order = 4 # order of the frame field (4 by default)

        # attributes (/!\ on work_mesh)
        self.curvature : M.Attribute = None # gaussian curvature at vertices
        self.barycenters : M.Attribute = None # barycenters of triangles
        self.areas : M.Attribute = None # areas of triangles
        self.edge_lengths : np.ndarray = None # length of all edges
        self.vnormals : M.Attribute = None # vector normals
        self.angles : M.Attribute = None # angle at each triangle corners
        self.defect : M.Attribute = None # sum of angles around a vertex (/!\ not 'real' defect which is 2*pi - this)
        self.connection : M.processing.SurfaceConnectionVertices = None

        self.PT_array : np.ndarray = None # parallel transport as a numpy array

        self._singular_faces : M.Attribute = None # boolean flag on faces. Accessed via a property
        self.singular_vertices : M.Attribute = None # boolean flag on vertices

        # Distortion and barriers
        self.init_var : np.ndarray = None # copy of self.var before optimizing. Kept as a reference.
        self.ref_dets : np.ndarray = None
        self.dist_matrices : np.ndarray = None

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

        self.edge_indices : np.ndarray = None # indices necessary for edge constraint energy
        self.rotFF_indices : np.ndarray = None # indices necessary for rot_follow_FF energy
        self.quad_indices : np.ndarray = None

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
            if self.order == 1 : return self._singular_faces
            
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
            if id_vertex >= len(self.connection._baseX): continue
            E,N = self.connection.base(id_vertex)[0], self.vnormals[id_vertex]
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