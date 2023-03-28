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
        self.parallel_transport : np.ndarray = None
        self._singular_vertices : M.Attribute = None
        self._singular_vertices_ff : M.attributes = None
        self.ref_edges : np.ndarray = None
        self.cotan_weight : np.ndarray = None
        self.rigid_matrices : np.ndarray = None

        # Variables and indirections
        self.var : np.ndarray = None # array of variables to optimize
        self.nvar : int = 0
        self.var_sep_rot : int = None # separator for easy variable access
        self.var_sep_ff : int = None # separator for easy variable access        
        self.var_sep_scale : int = None
        self.var_sep_dist : int = None

        # Indices
        self.edge_indices : np.ndarray = None
        self.ff_indices : np.ndarray = None
        self.rigid_ref_edges : np.ndarray = None

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
                for e in self.mesh.connectivity.vertex_to_edge(v):
                    u = self.mesh.connectivity.other_edge_end(e,v)
                    w = 2*atan(self.get_var_rot(e))
                    angle += w if v<u else -w
                if abs(angle)>ZERO_THRESHOLD:
                    self._singular_vertices[v] = angle*2/pi
            
            ff = M.processing.SurfaceFrameField(self.mesh,"faces", custom_connection=self.connection)
            ff.initialize()
            for T in self.mesh.id_faces:
                fft = self.get_var_ff(T)
                ff.var[T] = complex(fft[0], fft[1])
            ff.flag_singularities("singulsFF")
        return self._singular_vertices

    def export_frame_field(self):
        FFMesh = M.mesh.new_polyline()
        L = M.attributes.mean_edge_length(self.mesh)/3
        for id_face, face in enumerate(self.mesh.faces):
            pA,pB,pC = (self.mesh.vertices[u] for u in face)
            X,Y = self.local_base(id_face)
            Z = geom.cross(X,Y)
            ff = self.get_var_ff(id_face)
            ff = complex(ff[0], ff[1])
            angle = cmath.phase(ff)/4
            bary = (pA+pB+pC)/3 # reference point for display
            r1,r2,r3,r4 = (geom.rotate_around_axis(X, Z, angle + k*math.pi/2) for k in range(4))
            p1,p2,p3,p4 = (bary + abs(ff)*L*r for r in (r1,r2,r3,r4))
            FFMesh.vertices += [bary, p1, p2, p3, p4]
            FFMesh.edges += [(5*id_face, 5*id_face+k) for k in range(1,5)]
        return FFMesh

    def construct_param(self):
        root = 0
        if self.tree is None:
            self.tree = M.processing.trees.FaceSpanningTree(self.mesh, starting_face=root)()
        self.UVs = self.mesh.face_corners.create_attribute("uv_coords", float, 2)

        def build_triangle(iT):
            pA,pB,pC = (self.mesh.vertices[_u] for _u in self.mesh.faces[iT])
            X,Y,_ = self.local_base(iT)
            qA = M.Vec.zeros(2)
            qB = M.Vec(X.dot(pB-pA), Y.dot(pB-pA))
            qC = M.Vec(X.dot(pC-pA), Y.dot(pC-pA))
            J = self.get_jac(iT).reshape((2,2)).T
            qB = M.Vec(J.dot(qB))
            qC = M.Vec(J.dot(qC))
            return qA,qB,qC
        
        def align_triangles(pA,pB, qA, qB, qC):
            target = M.Vec(pB - pA)

            # 1) Scaling to match edge length
            # q = M.Vec(qB - qA)
            # scale = math.sqrt(target.dot(target) / q.dot(q))
            # qB = qA + (qB-qA)*scale
            # qC = qA + (qC-qA)*scale

            # 2) translation to align point A
            translation = pA - qA
            qA += translation
            qB += translation
            qC += translation

            # 3) rotation around point A to align the point B
            q = M.Vec(qB - qA)
            rot = math.atan2(target.y, target.x) - math.atan2(q.y,q.x)
            rc, rs = math.cos(rot), math.sin(rot)

            q = M.Vec(qB - qA)
            qB.x, qB.y = qA.x + rc*q.x - rs*q.y , qA.y + rs*q.x + rc*q.y
            q = M.Vec(qC-qA)
            qC.x, qC.y = qA.x + rc*q.x - rs*q.y , qA.y + rs*q.x + rc*q.y

            return qA,qB,qC

        for T,parent in self.tree.traverse():
            if parent is None:
                # build the root
                pA,pB,pC = build_triangle(root)
                self.UVs[(root,0)] = pA
                self.UVs[(root,1)] = pB
                self.UVs[(root,2)] = pC
                continue

            A,B = self.mesh.half_edges.common_edge(T,parent) # common vertices
            iA,iB = (self.mesh.connectivity.in_face_index(parent,_u) for _u in (A,B))
            jA,jB = (self.mesh.connectivity.in_face_index(T,_u) for _u in (A,B))
            jC = 3-jA-jB
            
            pA,pB = self.UVs[(parent,iA)], self.UVs[(parent,iB)]
            q = build_triangle(T)
            qA,qB,qC = q[jA], q[jB], q[jC]
            qA,qB,qC = align_triangles(pA,pB,qA,qB,qC)

            for (j,q) in [(jA,qA), (jB,qB), (jC,qC)]:
                self.UVs[(T,j)] = q
        
        flat_mesh = M.mesh.new_surface()
        for T in self.mesh.id_faces:
            for i in range(3):
                flat_mesh.vertices.append(M.Vec(self.UVs[(T,i)].x, self.UVs[(T,i)].y, 0.))
            flat_mesh.faces.append((3*T,3*T+1,3*T+2))
        return flat_mesh