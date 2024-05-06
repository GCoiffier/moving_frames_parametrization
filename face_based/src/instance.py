import mouette as M
from mouette import geometry as geom
import numpy as np
import math
from math import pi, atan
import cmath

class Instance:

    def __init__(self, _mesh : M.mesh.SurfaceMesh):
        self.mesh : M.mesh.SurfaceMesh = _mesh
        self.disk_mesh : M.mesh.SurfaceMesh = None
        self.flat_mesh : M.mesh.SurfaceMesh = None
        self.seam_graph : M.mesh.PolyLine = None
        self.singu_ptcld : M.mesh.PointCloud = None

        self.feat : M.processing.FeatureEdgeDetector = None

        # Attributes on self.mesh
        self.defect : M.Attribute = None # angle defect on vertices
        self.face_normals : M.Attribute = None
        self.connection : M.processing.SurfaceConnectionFaces = None
        self.PT_array : np.ndarray = None
        self._singular_vertices : M.Attribute = None
        self._singular_vertices_ff : M.attributes = None
        self.ref_edges : np.ndarray = None

        # Variables and indirections
        self.var : np.ndarray = None # array of variables to optimize
        self.nvar : int = 0
        self.var_sep_rot : int = None # separator for easy variable access
        self.var_sep_ff : int = None # separator for easy variable access        

        # Indices
        self.edge_indices : np.ndarray = None
        self.ff_indices : np.ndarray = None

        # Reconstruction
        self.tree : M.processing.trees.FaceSpanningTree = None
        self.UVs : M.Attribute = None
        self.seams : set = None

        # flags
        self.initialized : bool = False
        self.reconstructed : bool = False

    def local_base(self,iT):
        return self.connection.base(iT)

    def get_var_ff(self,iT) -> np.ndarray:
        return self.var[self.var_sep_ff + 2*iT: self.var_sep_ff + 2*iT+2]

    def get_var_rot(self,ie) -> float:
        return self.var[self.var_sep_rot + ie]

    def get_jac(self,iT) -> np.ndarray:
        # a c
        # b d
        return self.var[4*iT:4*(iT+1)]

    @property
    def singular_vertices(self):
        if self._singular_vertices is None:
            # flag singularities
            ZERO_THRESHOLD = 0.99*pi/4
            self._singular_vertices = self.mesh.vertices.create_attribute("singuls", float)
            for v in self.mesh.id_vertices:
                angle = self.defect[v]
                for e in self.mesh.connectivity.vertex_to_edges(v):
                    u = self.mesh.connectivity.other_edge_end(e,v)
                    w = 2*atan(self.get_var_rot(e))
                    angle += w if v<u else -w
                if abs(angle)>ZERO_THRESHOLD:
                    self._singular_vertices[v] = angle*2/pi
            
            ff = M.framefield.SurfaceFrameField(self.mesh,"faces", custom_connection=self.connection)
            ff.initialize()
            for T in self.mesh.id_faces:
                fft = self.get_var_ff(T)
                ff.var[T] = complex(fft[0], fft[1])
            ff.flag_singularities("singulsFF")
        return self._singular_vertices