import mouette as M
from copy import deepcopy
from .common import *

class FeatureCutter(M.processing.FeatureEdgeDetector):
    """FeatureCutter

    Adds several functionalities on the FeatureEdgeDetector, mainly to be able to cut the feature edges in order to get another mesh.
    """

    def __init__(self, mesh : M.mesh.SurfaceMesh, only_border:bool = False, corner_order:int = 4, verbose:bool = True):
        """
        Parameters:
            mesh (M.mesh.SurfaceMesh): the input mesh
            only_border (bool, optional): Flag to ignore feature edge detection and only work with boundary edges. Defaults to False.
            verbose (bool, optional): Verbose mode. Defaults to True.
        """

        super().__init__(only_border, corner_order=corner_order, verbose=verbose)
        self.name = "FeatureCutter" # overwrites name of M.Logger

        self.input_mesh : M.mesh.SurfaceMesh = mesh
        self.cut_mesh : M.mesh.SurfaceMesh = None

        self.cut_vertices = dict() # v -> list of versions of this split vertex
        self.cut_edges    = dict() # e -> two split edges
        self.ref_vertex   = dict() # reversed self.cut_vertices dict
        self.ref_edge     = dict() # reversed self.cut_edges dict
    
        self.corners_no_cuts : M.Attribute = None
        self.features_edges_no_cuts : M.Attribute = None
        self.feature_vertices_no_cuts : M.Attribute = None
        self.feature_degrees_no_cuts : M.Attribute = None

        self.vnormals : M.Attribute = None
    
    ##### 
    
    @property
    def original(self):
        """
        Returns a FeatureEdgeDetector object containing features as if we did not cut
        """
        feat_detect = M.processing.FeatureEdgeDetector(self.only_border, self.flag_corners, self.corner_order, self.compute_feature_graph, self.verbose)
        feat_detect.feature_degrees = self.feature_degrees_no_cuts
        feat_detect.feature_edges = self.features_edges_no_cuts
        feat_detect.feature_vertices = self.feature_vertices_no_cuts
        feat_detect.local_feat_edges = self.local_feat_edges
        return feat_detect

    def __call__(self):
        return self.run()

    def run(self):
        
        self.log("Compute features on original mesh")
        super().run(self.input_mesh) # compute features on the original mesh
        
        self.feature_degrees_no_cuts = deepcopy(self.feature_degrees)
        self.features_edges_no_cuts = deepcopy(self.feature_edges)
        self.feature_vertices_no_cuts = deepcopy(self.feature_vertices)
        self.corners_no_cuts = deepcopy(self.corners)
        self.cut_mesh = M.mesh.RawMeshData( M.mesh.copy(self.input_mesh, copy_attributes=True)) # /!\ copy necessary otherwise cut_mesh is only a shallow copy

        if (not self.only_border and len(self.feature_edges) != len(self.input_mesh.boundary_edges)) :
            self.compute_feature_graph = False # no need to recompute it
            self.cut()
            self.cut_mesh = M.mesh.SurfaceMesh(self.cut_mesh)
            super().run(self.cut_mesh)
        else:
            self.cut_mesh = M.mesh.SurfaceMesh(self.cut_mesh)
        return self
    
    #####

    def cut(self):
        """
        Builds self.cut_mesh as a new mesh

        Raises:
            Exception: fails if 'detect' is not called before 'cut'.
        """
        if self.feature_vertices is None or self.feature_edges is None:
            raise Exception("[FeatureManager] Cannot cut before detecting")
        self.ref_vertex = dict([(v, None) for v in self.input_mesh.id_vertices])
        
        ring_degrees = dict() # the number of parts the ring of each vertex is cut into. Related to feature_degree but -1 on border
        for A in self.feature_degrees:
            if self.input_mesh.is_vertex_on_border(A):
                ring_degrees[A] = self.feature_degrees[A] - 1
            else:
                ring_degrees[A] = self.feature_degrees[A]
        
        # split every feature vertex according to its degree
        for ref_v in self.feature_vertices:
            self.ref_vertex[ref_v] = ref_v
            self.cut_vertices[ref_v] = [ref_v]
            pV = self.cut_mesh.vertices[ref_v]
            for _ in range(ring_degrees[ref_v]-1):
                ind = len(self.cut_mesh.vertices)
                self.cut_vertices[ref_v].append(ind)
                self.ref_vertex[ind] = ref_v
                self.cut_mesh.vertices.append(pV)

        # split rings of vertices
        for ref_v in self.feature_vertices:
            iring = 0
            deg = ring_degrees[ref_v]
            new_v = self.cut_vertices[ref_v][iring]
            for iF, F in enumerate(self.input_mesh.connectivity.vertex_to_face(ref_v)): # self.input_mesh instead of cutmesh so we do not recompute connectivity data just for this
                A,B,C = self.cut_mesh.faces[F]
                if iF in self.local_feat_edges[ref_v]:
                    iring += 1
                    new_v = self.cut_vertices[ref_v][iring%deg]
                self.cut_mesh.faces[F] = replace_in_list([A,B,C], ref_v, new_v) # replace v by iv in the face

        # recompute corners
        self.cut_mesh.face_corners.clear()
        self.cut_mesh.edges.clear()
            
        # compute self.cut_edges and orig_edge
        for iu, (A,B) in enumerate(self.cut_mesh.edges):
            if self.ref_vertex[A] is None or self.ref_vertex[B] is None: continue
            old_ie = self.input_mesh.connectivity.edge_id(self.ref_vertex[A], self.ref_vertex[B])
            self.ref_edge[iu] = old_ie
            if old_ie not in self.cut_edges:
                self.cut_edges[old_ie] = [iu]
            else:
                self.cut_edges[old_ie].append(iu)