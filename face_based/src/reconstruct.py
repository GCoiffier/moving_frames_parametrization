import mouette as M
from mouette.processing import SingularityCutter
from mouette import geometry as geom

from .worker import *
from .instance import Instance

import math
import cmath

def create_optimal_seams(I : Instance, features : bool, verbose) -> SingularityCutter:
    """Performs minimal set of cuts between singularities"""
    singu_set = {x for x in I.singular_vertices}
    if not features or (len(I.feat.feature_edges) == len(I.mesh.boundary_edges)):
        # no features detected (does not count boundary edges as features)
        cutter = SingularityCutter(I.mesh, singu_set, verbose=verbose)() 
    else:
        cutter = SingularityCutter(I.mesh, singu_set, features=I.feat, verbose=verbose)()
    I.seams = cutter.cut_edges
    return cutter

def rescale_uvs(I : Instance):
    """Scales I.UVs in bounding box [0;1]^2"""
    xmin,xmax,ymin,ymax = float("inf"), -float("inf"), float("inf"), -float("inf")
    for T in I.mesh.id_faces:
        for i in range(3):
            uv = I.UVs[3*T+i]
            xmin = min(xmin, uv.x)
            xmax = max(xmax, uv.x)
            ymin = min(ymin, uv.y)
            ymax = max(ymax, uv.y)
    scale_x = xmax-xmin
    scale_y = ymax-ymin
    scale = min(scale_x, scale_y)

    # rotate I.UVs so that feature edges are axis aligned
    if I.feat.feature_edges:
        e = list(I.feat.feature_edges)[0]
        A,B = I.mesh.edges[e]
        T, iA,iB = I.mesh.connectivity.direct_face(A,B,True)
        if T is None:
            T,iB,iA = I.mesh.connectivity.direct_face(B,A,True)
        vec = I.UVs[3*T+iB] - I.UVs[3*T+iA]
        angle = -math.atan2(vec.y, vec.x)
    else:
        angle = 0.

    # apply transformation
    for T in I.mesh.id_faces:
        for i in range(3):
            I.UVs[3*T+i] = geom.rotate_2d(I.UVs[3*T+i] / scale, angle)
    return I.UVs

def write_output_obj(I : Instance, file_path : str):
    """Final export of the mesh as an obj file with custom fields for singularity cones, seams and feature edges"""
    M.mesh.save(I.mesh, file_path)
    # now export cones, seams and features as special fields in .obj
    with open(file_path, "a") as fr:
        for s in I.singular_vertices:
            idx = round(I.singular_vertices[s])
            if idx==-1:
                fr.write(f"c {s+1} -1\n")
            elif idx==1:
                fr.write(f"c {s+1} 1\n")
            else:
                fr.write(f"c {s+1} 0\n")
        
        for e in I.seams:
            a,b = I.mesh.edges[e]
            fr.write(f"sm {a+1} {b+1}\n")

        for e in I.feat.feature_edges:
            a,b = I.mesh.edges[e]
            if not I.mesh.is_edge_on_border(a,b):
                fr.write(f"ft {a+1} {b+1}\n")

class ParamConstructor(Worker):
    """Worker responsible for putting the parametrization back together after optimization. Also exports various debug outputs"""

    def __init__(self, instance: Instance, options : Options, verbose_options : VerboseOptions):
        super().__init__("ParamReconstruction", instance, options, verbose_options)
        self.cutter : SingularityCutter = None

    def __call__(self):
        self.construct_param()
        return self

    def export_frame_field(self) -> M.mesh.PolyLine:
        I = self.instance
        FFMesh = M.mesh.PolyLine()
        L = M.attributes.mean_edge_length(I.mesh)/3
        for id_face, face in enumerate(I.mesh.faces):
            pA,pB,pC = (I.mesh.vertices[u] for u in face)
            X,Y = I.local_base(id_face)
            Z = geom.cross(X,Y)
            ff = I.get_var_ff(id_face)
            ff = complex(ff[0], ff[1])
            angle = cmath.phase(ff)/4
            bary = (pA+pB+pC)/3 # reference point for display
            r1,r2,r3,r4 = (geom.rotate_around_axis(X, Z, angle + k*math.pi/2) for k in range(4))
            p1,p2,p3,p4 = (bary + abs(ff)*L*r for r in (r1,r2,r3,r4))
            FFMesh.vertices += [bary, p1, p2, p3, p4]
            FFMesh.edges += [(5*id_face, 5*id_face+k) for k in range(1,5)]
        return FFMesh

    def export_feature_graph(self) -> M.mesh.PolyLine:
       return self.instance.feat.feature_graph

    def export_seams(self) -> M.mesh.PolyLine:
        if not self.instance.reconstructed : 
            self.log("Instance UVs were not reconstructed. Call `construct_param()` before this for a result != None")
            return None
        return self.instance.seam_graph

    def export_singularity_point_cloud(self) -> M.mesh.PointCloud:
        I = self.instance
        I.singu_ptcld = M.mesh.PointCloud()
        index = I.singu_ptcld.vertices.create_attribute("index", int)
        i = 0
        for iV in I.singular_vertices:
            if abs(I.singular_vertices[iV])<1e-8: continue
            I.singu_ptcld.vertices.append(I.mesh.vertices[iV])
            index[i] = round(I.singular_vertices[iV])
            i += 1
        for v in I.feat.feature_vertices:
            if I.feat.corners[v] in (2,4) : continue
            I.singu_ptcld.vertices.append(I.mesh.vertices[v])
            index[i] = I.feat.corners[v]
            i += 1
        return I.singu_ptcld

    def export_flat_mesh(self) -> M.mesh.SurfaceMesh:
        """Builds the parametrization as if (x,y) = (u,v)"""
        if not self.instance.reconstructed : 
            self.log("Instance UVs were not reconstructed. Call `construct_param()` before this for a result != None")
            return None
        I = self.instance
        I.flat_mesh = M.mesh.RawMeshData()
        for T in I.mesh.id_faces:
            for i in range(3):
                I.flat_mesh.vertices.append(M.Vec(I.UVs[3*T+i].x, I.UVs[3*T+i].y, 0.))
            I.flat_mesh.faces.append((3*T,3*T+1,3*T+2))
        return M.mesh.SurfaceMesh(I.flat_mesh)

    def export_disk_mesh(self):
        """Input mesh but with a disk topology, where seams are real cuts"""
        I = self.instance
        I.disk_mesh = M.mesh.copy(self.cutter.output_mesh)
        UVcut = I.disk_mesh.face_corners.create_attribute("uv_coords",float,2)
        for T in I.mesh.id_faces:
            for i in range(3):
                UVcut[3*T+i] = I.UVs[3*T+i]
        return I.disk_mesh

    def construct_param(self):
        I = self.instance
        self.cutter = create_optimal_seams(I, self.options.features, self.verbose_options.logger_verbose)

        self.log("Starting UV reconstruction.")
        root = 0
        I.tree = M.processing.trees.FaceSpanningTree(I.mesh, starting_face=root, forbidden_edges=I.seams)()
        I.UVs = I.mesh.face_corners.create_attribute("uv_coords", float, 2)
        
        def build_triangle(iT):
            pA,pB,pC = (I.mesh.vertices[_u] for _u in I.mesh.faces[iT])
            X,Y = I.local_base(iT)
            qA = M.Vec.zeros(2)
            qB = M.Vec(X.dot(pB-pA), Y.dot(pB-pA))
            qC = M.Vec(X.dot(pC-pA), Y.dot(pC-pA))
            J = I.get_jac(iT).reshape((2,2)).T
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

        self.log("Traverse face tree")
        for T,parent in I.tree.traverse():
            if parent is None:
                # build the root
                pA,pB,pC = build_triangle(root)
                I.UVs[3*root+0] = pA
                I.UVs[3*root+1] = pB
                I.UVs[3*root+2] = pC
                continue

            A,B = I.mesh.connectivity.common_edge(T,parent) # common vertices
            iA,iB = (I.mesh.connectivity.in_face_index(parent,_u) for _u in (A,B))
            jA,jB = (I.mesh.connectivity.in_face_index(T,_u) for _u in (A,B))
            jC = 3-jA-jB
            
            pA,pB = I.UVs[3*parent+iA], I.UVs[3*parent+iB]
            q = build_triangle(T)
            qA,qB,qC = q[jA], q[jB], q[jC]
            qA,qB,qC = align_triangles(pA,pB,qA,qB,qC)

            for (j,q) in [(jA,qA), (jB,qB), (jC,qC)]:
                I.UVs[3*T+j] = q
        self.log("Tree traversal done.")

        self.log("Scaling and alignement with axes")
        I.UVs = rescale_uvs(I)

        I.reconstructed = True
        I.seam_graph = self.cutter.cut_graph