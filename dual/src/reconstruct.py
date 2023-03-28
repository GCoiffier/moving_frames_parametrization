import math
import numpy as np
from collections import deque

import mouette as M
import mouette.geometry as geom
from mouette.geometry import rotate_2d
from mouette.processing import SingularityCutter, SurfaceSubdivision

from .common import *
from .instance import Instance
from .worker import *

##### Utility functions #####

def split_singular_triangles(I : Instance):
    """Combinatorial operations on the mesh to place additionnal points inside singular triangles"""
    I.triplets_of_triangles = dict() # singular triangles are cut in 3 parts

    I.singu_faces_to_4pts = dict()
    I.triangles_before_split = dict()
    I.singular_vertices = I.mesh.vertices.create_attribute("singularities", int)
    if len(I.singular_faces)==0 : return I

    nF = len(I.mesh.faces)
    with SurfaceSubdivision(I.work_mesh) as subdiv1:
        with SurfaceSubdivision(I.mesh) as subdiv2:
            for f in range(nF):
                if I.singular_faces[f] == 0 : continue
                # split triangle in 3
                A,B,C = I.mesh.faces[f]
                nV,nF = len(I.mesh.vertices), len(I.mesh.faces)
                I.singular_vertices[nV] = I.singular_faces[f]
                # split function adds a new vertex (the face barycenter) at the end 
                subdiv1.split_face_as_fan(f)
                subdiv2.split_face_as_fan(f)
                I.singu_faces_to_4pts[f] = [nV,A,B,C]
                I.triplets_of_triangles[f] = (f, nF, nF+1)
                I.triplets_of_triangles[nF] = (nF, nF+1, f)
                I.triplets_of_triangles[nF+1] = (nF+1, f, nF)
                I.triangles_before_split[f] = (f,(A,B,C))
                I.triangles_before_split[nF] = (f,(A,B,C))
                I.triangles_before_split[nF+1] = (f,(A,B,C))
    # Resets invalid connectivity
    # This shouldn't be necessary but better be explicit
    I.mesh.connectivity.clear()
    I.mesh.half_edges.clear()
    I.work_mesh.connectivity.clear()
    I.work_mesh.half_edges.clear()
    return I

def replace_singularities_conformal_and_auth(I : Instance) -> int:
    """Replacing singularities minimizing a compromise between a conformal distortion ||J||^2 / det J and area distortion det J + 1/ det J

    Returns:
        int: number of singularities for which replacement has failed
    """
    raise NotImplementedError
    a = 0.5 # weight balance between the two terms
    n_singus_fail = 0
    for singuTri in I.singular_faces:
        try:
            S,A,B,C = I.singu_faces_to_4pts[singuTri]
            pA,pB,pC = (I.mesh.vertices[_u] for _u in (A,B,C))
            X = M.Vec.normalized(pB-pA)
            Y = M.Vec.normalized(pC-pA)
            Z = geom.cross(X,Y)
            mat = np.zeros((2,2))
            rhs = M.Vec.zeros(2)
            for ref in (A,B,C):
                T, iref, iS = I.mesh.half_edges.adj(ref,S)
                inxt = 3-iref-iS
                nxt = I.mesh.faces[T][inxt]
                uv_ref, uv_nxt, uv_s = (M.Vec(I.UVs[(T,_i)].x, I.UVs[(T,_i)].y) for _i in (iref,inxt,iS))

                # in parametric domain
                UV = np.array([uv_nxt - uv_ref, uv_s - uv_ref])
                det_uv = np.linalg.det(UV)
                G = np.linalg.inv(UV.T @ UV)

                # in real domain (basis of triangle)
                Pref = I.mesh.vertices[ref]
                Pref = M.Vec(X.dot(Pref), Y.dot(Pref))
                Pnxt = I.mesh.vertices[nxt]
                Pnxt = M.Vec(X.dot(Pnxt), Y.dot(Pnxt))
                P = Pnxt-Pref
                U = np.array([[ P.y*P.y, -P.x*P.y   ], [-P.x*P.y, P.x*P.x ]])
                mat_ref = a*det_uv*G + (1-a)/det_uv * U
                mat += mat_ref
                rhs += mat_ref.dot(Pref)

            pS = np.linalg.solve(mat, rhs)
            # change of basis
            basisT = np.array((X,Y,Z))
            basisT = np.linalg.inv(basisT)
            I.mesh.vertices[S] = basisT.dot( M.Vec(pS.x, pS.y, Z.dot((pA+pB+pC)/3)))
        except np.linalg.LinAlgError as e:
            n_singus_fail += 1

    return n_singus_fail

def replace_singularities_ARAP_Jac(I : Instance):
    """Replacing singularities minimizing the ARAP energy, ie the three jacobian of adjacent triangles of a singularity should be isometries

    Returns:
        int: number of singularities for which replacement has failed
    """

    n_singus_fail = 0
    for singuTri in I.singular_faces:
        try:
            # Get coordinates in x,y space
            S,A,B,C = I.singu_faces_to_4pts[singuTri]

            pA,pB,pC,pS = (I.mesh.vertices[_u] for _u in (A,B,C,S))
            X,Y,Z = geom.face_basis(pA,pB,pC)
            pA, pB, pC, pS = (M.Vec( X.dot(_p), Y.dot(_p) ) for _p in (pA,pB,pC,pS)) # project in basis of the triangle

            # Get coordinates in u,v space
            T1, iA1, iS1 = I.mesh.half_edges.adj(A,S) # T1 = ACS
            iC1 = 3 - iA1 - iS1
            cnr = I.mesh.connectivity.face_to_first_corner(T1)
            uA1, uC1, uS1 = ( M.Vec(I.UVs[cnr + _i].x, I.UVs[cnr + _i].y) for _i in (iA1,iC1,iS1))
            area1 = geom.triangle_area_2D(uA1,uC1,uS1)

            T2, iB2, iS2 = I.mesh.half_edges.adj(B,S) # T2 = ABS
            iA2 = 3 - iB2 - iS2
            cnr = I.mesh.connectivity.face_to_first_corner(T2)
            uA2, uB2, uS2 = ( M.Vec(I.UVs[cnr + _i].x, I.UVs[cnr + _i].y) for _i in (iA2,iB2,iS2))
            area2 = geom.triangle_area_2D(uA2,uB2,uS2)

            T3, iC3, iS3 = I.mesh.half_edges.adj(C,S) # T3 = BCS
            iB3 = 3 - iC3 - iS3
            cnr = I.mesh.connectivity.face_to_first_corner(T3)
            uB3, uC3, uS3 = ( M.Vec(I.UVs[cnr + _i].x, I.UVs[cnr + _i].y) for _i in (iB3,iC3,iS3))
            area3 = geom.triangle_area_2D(uB3,uC3,uS3)

            # build system
            J1 = np.array([uC1 - uS1, uA1 - uS1]).T
            J2 = np.array([uA2 - uS2, uB2 - uS2]).T
            J3 = np.array([uB3 - uS3, uC3 - uS3]).T
            J1,J2,J3 = (np.linalg.inv(_J) for _J in (J1,J2,J3))

            R1,R2,R3 = np.eye(2), np.eye(2), np.eye(2)
            systm = np.zeros((12,2))

            for iter in range(100): # alternate solve for rotations and solve for S
                # First solve for rotations
                S1 = np.array([pC-pS, pA-pS]).T @ J1
                U,L,V = np.linalg.svd(S1)
                R1 = (V @ U).T
                if np.linalg.det(R1) < 0:
                    U[:,1] *= -1
                    R1 = (V @ U).T

                S2 = np.array([pA-pS, pB-pS]).T @ J2
                U,L,V = np.linalg.svd(S2)
                R2 = (V @ U).T
                if np.linalg.det(R2) < 0:
                    U[:,1] *= -1
                    R2 = (V @ U).T

                S3 = np.array([pB-pS, pC-pS]).T @ J3
                U,L,V = np.linalg.svd(S3)
                R3 = (V @ U).T
                if np.linalg.det(R3) < 0:
                    U[:,1] *= -1
                    R3 = (V @ U).T
            
                # now solve for S
                rhs = np.zeros(12)
                for k,(ar, J,R, P1,P2) in enumerate([(area1, J1,R1, pC, pA), (area2, J2,R2, pA, pB), (area3, J3,R3, pB, pC)]):
                    systm[4*k+0,0] = (J[0,0] + J[1,0])
                    systm[4*k+1,0] = (J[0,1] + J[1,1])
                    systm[4*k+2,1] = (J[0,0] + J[1,0])
                    systm[4*k+3,1] = (J[0,1] + J[1,1])
                    r = np.array([P1,P2]).T @ J - R
                    rhs[4*k+0] = r[0,0]
                    rhs[4*k+1] = r[0,1]
                    rhs[4*k+2] = r[1,0]
                    rhs[4*k+3] = r[1,1]
                newS = np.linalg.lstsq(systm, rhs, rcond=None)[0]

                if M.geometry.distance(newS,pS)<1e-6:
                    if geom.det_2x2(pA-pS,pB-pS)<0 or geom.det_2x2(pB-pS,pC-pS)<0 or geom.det_2x2(pC-pS,pA-pS)<0 :
                        pS = (pA+pB+pC)/3
                        n_singus_fail += 1
                        print(pA,pB,pC)
                        print(uA1,uC1,uS1)
                        print(uA2,uB2,uS2)
                        print(uB3,uC3,uS3)
                        print()
                    break
                pS = M.Vec(newS)


            basisT = np.array((X,Y,Z))
            basisT = np.linalg.inv(basisT)
            bary = sum(I.mesh.vertices[_u] for _u in (A,B,C))/3
            I.mesh.vertices[S] = basisT.dot( M.Vec(pS.x, pS.y, Z.dot(bary)))


        except Exception as e:
            # print(e)
            n_singus_fail += 1
    return n_singus_fail

def replace_singularities_barycenter(I : Instance):
    """Replacing singularities minimizing the ARAP energy, ie the three jacobian of adjacent triangles of a singularity should be isometries

    Returns:
        int: number of singularities for which replacement has failed
    """

    n_singus_fail = 0
    for singuTri in I.singular_faces:
        try:
            # Get coordinates in x,y space
            S,A,B,C = I.singu_faces_to_4pts[singuTri]

            pA,pB,pC,pS = (I.mesh.vertices[_u] for _u in (A,B,C,S))
            X,Y,Z = geom.face_basis(pA,pB,pC)
            pA, pB, pC, pS = (M.Vec( X.dot(_p), Y.dot(_p) ) for _p in (pA,pB,pC,pS)) # project in basis of the triangle

            # Get coordinates in u,v space
            T1, iA1, iS1 = I.mesh.half_edges.adj(A,S) # T1 = ACS
            iC1 = 3 - iA1 - iS1
            cnr = I.mesh.connectivity.face_to_first_corner(T1)
            uA1, uC1, uS1 = ( M.Vec(I.UVs[cnr + _i].x, I.UVs[cnr + _i].y) for _i in (iA1,iC1,iS1))
            area1 = geom.triangle_area_2D(uA1,uC1,uS1)

            T2, iB2, iS2 = I.mesh.half_edges.adj(B,S) # T2 = ABS
            iA2 = 3 - iB2 - iS2
            cnr = I.mesh.connectivity.face_to_first_corner(T2)
            uA2, uB2, uS2 = ( M.Vec(I.UVs[cnr + _i].x, I.UVs[cnr + _i].y) for _i in (iA2,iB2,iS2))
            area2 = geom.triangle_area_2D(uA2,uB2,uS2)

            T3, iC3, iS3 = I.mesh.half_edges.adj(C,S) # T3 = BCS
            iB3 = 3 - iC3 - iS3
            cnr = I.mesh.connectivity.face_to_first_corner(T3)
            uB3, uC3, uS3 = ( M.Vec(I.UVs[cnr + _i].x, I.UVs[cnr + _i].y) for _i in (iB3,iC3,iS3))
            area3 = geom.triangle_area_2D(uB3,uC3,uS3)
            
            tot_area = area1 + area2 + area3
            pS = (area3*pA + area2*pC + area1*pB) / tot_area

            basisT = np.array((X,Y,Z))
            basisT = np.linalg.inv(basisT)
            bary = sum(I.mesh.vertices[_u] for _u in (A,B,C))/3
            I.mesh.vertices[S] = basisT.dot( M.Vec(pS.x, pS.y, Z.dot(bary)))

        except Exception as e:
            # print(e)
            n_singus_fail += 1
    return n_singus_fail

def create_optimal_seams(I : Instance, features : bool, verbose) -> SingularityCutter:
    """Performs minimal set of cuts between singularities"""
    singu_set = {x for x in I.singular_vertices}
    if not features or (len(I.feat.feature_edges) == len(I.mesh.boundary_edges)):
        # no features detected
        cutter = SingularityCutter(I.mesh, singu_set, verbose=verbose)() 
    else:
        featdetect = I.feat.original # extract detection from cutter
        cutter = SingularityCutter(I.mesh, singu_set, features=featdetect, verbose=verbose)()
    I.seams = cutter.cut_edges
    return cutter

def delimit_feature_regions(I : Instance, cutter ) -> M.utils.UnionFind:
    """When dealing with feature edges, the tree traversal for reconstruction needs to be modified. 
    Features delimit regions inside the mesh, that should only be connected by one edge.
    If such a region is reconstructed from two or more edges, new seams appear due to period jumps along existing seams.
    Taking this into account is a pain, so we instead flag everything for the traversal to avoid such cases.

    Returns:
        An Union-Find structure that tells us in which component each triangle is.
    """
    triangle_region = M.utils.UnionFind(I.mesh.id_faces)
    forbidden_edges_set = cutter.cut_edges | I.feat.features_edges_no_cuts
    forbidden_edges = M.Attribute(bool) #self.input_mesh.edges.create_attribute("forbidden_edges", bool)
    for e in forbidden_edges_set:
        forbidden_edges[e] = True

    I.tree = M.processing.trees.FaceSpanningForest(I.mesh, forbidden_edges_set)()
    for vertex,father in I.tree.traverse():
        if father is not None:
            triangle_region.union(vertex,father)
    return triangle_region

def rescale_uvs(I : Instance):
    """Scales I.UVs in bounding box [0;1]^2"""
    xmin,xmax,ymin,ymax = float("inf"), -float("inf"), float("inf"), -float("inf")
    for c in I.mesh.id_corners:
        uv = I.UVs[c]
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
        A,B = I.work_mesh.edges[e]
        T, iA,iB = I.work_mesh.half_edges.adj(A,B)
        if T is None:
            T,iB,iA = I.work_mesh.half_edges.adj(B,A)
        cnr = I.work_mesh.connectivity.face_to_first_corner(T)
        vec = I.UVs[cnr+ iB] - I.UVs[cnr + iA]
        angle = -atan2(vec.y, vec.x)
    else:
        ref_frame = complex(I.var[I.var_sep_ff], I.var[I.var_sep_ff+1])
        angle = cmath.phase(ref_frame)/I.order

    # apply transformation
    for c in I.mesh.id_corners:
        I.UVs[c] = rotate_2d(I.UVs[c] / scale, angle)
        # I.UVs[c] = I.UVs[c] / scale
    return I.UVs

def write_output_obj(I : Instance, file_path : str):
    """Final export of the mesh as an obj file with custom fields for singularity cones, seams and feature edges"""
    M.mesh.save(I.mesh, file_path)
    # now export cones, seams and features as special fields in .obj
    with open(file_path, "a") as fr:
        for s in I.singular_vertices:
            idx = I.singular_vertices[s]
            if idx==-1:
                fr.write(f"c {s+1} -1\n")
            elif idx==1:
                fr.write(f"c {s+1} 1\n")
            else:
                fr.write(f"c {s+1} 0\n")
        
        for e in I.seams:
            a,b = I.mesh.edges[e]
            fr.write(f"sm {a+1} {b+1}\n")

        for e in I.feat.features_edges_no_cuts:
            a,b = I.mesh.edges[e]
            if not I.mesh.is_edge_on_border(a,b):
                fr.write(f"ft {a+1} {b+1}\n")


class ParamConstructor(Worker):
    """Worker responsible for putting the parametrization back together after optimization. Also exports various debug outputs"""

    def __init__(self, instance: Instance, options = Options(), verbose_options = VerboseOptions()):
        super().__init__("ParamReconstruction", instance, options, verbose_options)
        self.cutter : SingularityCutter = None
        self.reconstructed : bool = False

    def __call__(self):
        self.construct_param()
        return self

    def export_frame_field(self) -> M.mesh.PolyLine:
        """
        Exports the frame field as a mesh for visualization.

        Returns:
            PolyLine: the frame field as a mesh object
        """
        return self.instance.export_frame_field()

    def export_feature_graph(self) -> M.mesh.PolyLine:
       return self.instance.feat.feature_graph

    def export_seams(self) -> M.mesh.PolyLine:
        if not self.reconstructed : return None
        return self.instance.cut_graph

    def export_singularity_point_cloud(self) -> M.mesh.PointCloud:
        I = self.instance
        I.singu_ptcld = M.mesh.new_point_cloud()
        index = I.singu_ptcld.vertices.create_attribute("index", float)
        i = 0
        for iF in I.work_mesh.id_faces:
            if abs(I.singular_faces[iF])>1e-8:
                P = I.singu_faces_to_4pts[iF][0]
                I.singu_ptcld.vertices.append(I.mesh.vertices[P])
                index[i] = I.singular_faces[iF]
                i += 1
        for v in I.feat.corners:
            if I.feat.corners[v] != I.order//2 :
                I.singu_ptcld.vertices.append(I.work_mesh.vertices[v])
                index[i] = I.feat.corners_no_cuts[v]
                i += 1
        return I.singu_ptcld

    def export_flat_mesh(self) -> M.mesh.SurfaceMesh:
        """Builds the parametrization as if (x,y) = (u,v)"""
        if not self.reconstructed : return None 
        I = self.instance
        I.param_mesh = M.mesh.new_surface()
        for T in I.mesh.id_faces:
            cnr = I.mesh.connectivity.face_to_first_corner(T)
            for i in range(3):
                I.param_mesh.vertices.append(M.Vec(I.UVs[cnr + i].x, I.UVs[cnr + i].y, 0.))
            I.param_mesh.faces.append((3*T,3*T+1,3*T+2))
            I.param_mesh.face_corners += [3*T,3*T+1,3*T+2]
        singular_tri = I.param_mesh.faces.create_attribute("singular", bool)
        for f in I.triplets_of_triangles:
            singular_tri[f] = True
        return I.param_mesh

    def export_disk_mesh(self):
        """Input mesh but with a disk topology, where seams are real cuts"""
        I = self.instance
        I.disk_mesh = M.mesh.copy(self.cutter.output_mesh)
        UVcut = I.disk_mesh.face_corners.create_attribute("uv_coords",float,2)
        for c in I.mesh.id_corners:
            UVcut[c] = I.UVs[c]
        return I.disk_mesh

    def construct_param(self):
        I = self.instance

        I = split_singular_triangles(I) # Also resets connectivity if needed
        self.cutter = create_optimal_seams(I, self.options.features, self.verbose_options.logger_verbose)         
        self.log("Starting UV reconstruction.")
        # We reconstruct along a spanning tree whose root is not singular
        root = 0
        while root in I.triplets_of_triangles: # root is singular
            root += 1

        visited = M.ArrayAttribute(bool, len(I.mesh.faces)) #  I.mesh.faces.create_attribute("visited", bool, dense=True)
        I.UVs = I.mesh.face_corners.create_attribute("uv_coords", float, 2)
        queue = deque()

        I.barycenters = M.attributes.face_barycenter(I.mesh, persistent=False) # recompute barycenters since we have split some triangles

        triangle_region = delimit_feature_regions(I, self.cutter) # Compute regions delimited by features edges

        def build_edge(A,B):
            ie = I.work_mesh.connectivity.edge_id(A,B)
            direct = (A<B)
            we = I.get_rotation(ie) # angles defined on cut mesh 
            # edge A -> m_AB -> B
            imA, imB = 4*ie, 4*ie + 2
            if not direct:
                imA,imB = imB,imA
                we *= -1
            mA, mB = complex(I.var[imA], I.var[imA+1]), complex(I.var[imB], I.var[imB+1])
            wpt = I.PT_array[ie] if direct else - I.PT_array[ie]
            rotB = principal_angle(we - wpt + pi)
            return c3vec(mA - mB*crot(rotB))

        def build_edge_to_center(T, A):
            Torig,(Ao,Bo,Co) = I.triangles_before_split[T]
            iA = np.argmax([Ao==A, Bo==A, Co==A]) # index of A in original triangle
            iS = I.var_sep_pt + 6*Torig + 2*iA
            return M.Vec(I.var[iS], I.var[iS+1], 0.)

        def build_triangle(T):
            A,B,C = I.work_mesh.faces[T]
            pA = M.Vec.zeros(3)
            pB = build_edge(A,B)
            pC = build_edge(A,C)
            return pA,pB,pC

        def build_triangle_singular(T, iA, iB):
            A,B = (I.work_mesh.faces[T][_x] for _x in (iA,iB))
            pA = M.Vec.zeros(3)
            pB = build_edge(A,B)
            pS = build_edge_to_center(T,A)
            return pA,pB,pS

        def can_traverse(a,b, T1=None,T2=None):
            e = I.mesh.connectivity.edge_id(a,b)
            if e in self.cutter.cut_edges: 
                return False
            if T1 is None or T2 is None:
                T1,T2 = I.mesh.half_edges.edge_to_triangles(a,b)
            if T1 is None or T2 is None: 
                return False # edge on border -> no need to push in queues
            is_feat = (e in I.feat.features_edges_no_cuts)
            return not (is_feat and triangle_region.connected(T1,T2))

        def push(T, T2, A, iAT, pA, B, iBT, pB):
            if (T2 is not None) and (not visited[T2]) and can_traverse(A,B, T, T2):
                triangle_region.union(T,T2)
                queue.append((T,T2,iAT,pA,iBT,pB))

        # Build the root triangle
        A,B,C = I.mesh.faces[root]
        pA,pB,pC = build_triangle(root)
        cnr = I.mesh.connectivity.face_to_first_corner(root)
        I.UVs[cnr] = pA.xy
        I.UVs[cnr + 1] = pB.xy
        I.UVs[cnr + 2] = pC.xy
        visited[root] = True

        # append adjacent triangles
        TAB, iA, iB = I.mesh.half_edges.opposite(A, B, root)
        push(root, TAB, A, iA, pA, B, iB, pB)

        TBC, iB, iC = I.mesh.half_edges.opposite(B, C, root)
        push(root, TBC, B, iB, pB, C, iC, pC)

        TCA, iC, iA = I.mesh.half_edges.opposite(C, A, root)
        push(root, TCA, C, iC, pC, A, iA, pA)
        
        # traverse the face tree
        self.log("Traverse face tree")

        while len(queue)>0: 
            father, T, iAT, pA, iBT, pB = queue.popleft()
            if T is None: continue # edge was on border -> nothing on the other side
            if visited[T] : continue
            visited[T] = True

            A,B = (I.mesh.ith_vertex_of_face(T,_u) for _u in (iAT,iBT))

            # build third vertex C from to origins A and B following edges
            if T in I.triplets_of_triangles: # T is part of a singular triangle
                # -> we build the three triangles in the triplet.

                # 1) Build singularity position (not using build_edge function since construction is different)
                iST = 3 - iAT - iBT
                S = I.mesh.ith_vertex_of_face(T,iST)
                qA,qB,qS = build_triangle_singular(T,iAT,iBT)
                qA,qB,qS = align_edges(pA,pB,qA,qB,qS)
                cnr = I.mesh.connectivity.face_to_first_corner(T)
                I.UVs[cnr + iAT] = qA.xy
                I.UVs[cnr + iBT] = qB.xy
                I.UVs[cnr + iST] = qS.xy

                # 2) build two other triangles
                # there are 2 cases whether we are adjacent to the cut or not
                T_on_cut = not ( can_traverse(A,S) and can_traverse(B,S) )
                pA,pB,pS = qA,qB,qS # ref for later

                if T_on_cut : # build T1 and T2 in consistent order
                    if I.mesh.half_edges.opposite(B,S,T)[0] is None or (not can_traverse(B,S)): # triangles are on the other side but we take B as a notation
                        B, pB = A, pA
                    T1, iBT1, iST1 = I.mesh.half_edges.opposite(B,S,T)
                    if T1 is None or (not can_traverse(B,S)): continue # we cannot traverse on both side -> stop here

                    visited[T1] = True
                    iCT1 = 3 - iBT1 - iST1
                    # build C from known points
                    C = I.mesh.ith_vertex_of_face(T1, iCT1)
                    qB,qC,qS = build_triangle_singular(T1, iBT1, iCT1)
                    qB,qS,qC = align_edges(pB,pS,qB,qS,qC) # /!\ known points to align are B and S, not B and C

                    cnr = I.mesh.connectivity.face_to_first_corner(T1)
                    I.UVs[cnr + iBT1] = qB.xy
                    I.UVs[cnr + iCT1] = qC.xy
                    I.UVs[cnr + iST1] = qS.xy

                    T1n, iBT1, iCT1 = I.mesh.half_edges.opposite(B,C,T1)
                    push(T1, T1n, B, iBT1, qB, C, iCT1, qC)

                    # T2
                    pS,pC = qS,qC
                    T2, iCT2, iST2 = I.mesh.half_edges.opposite(C,S,T1)
                    if T2 is None or (not can_traverse(C,S)): continue # cannot access third triangle -> stop here
                    visited[T2] = True
                    iDT2 = 3 - iCT2 - iST2
                    D = I.mesh.ith_vertex_of_face(T2, iDT2)
                    qC,qD,qS = build_triangle_singular(T2,iCT2,iDT2)
                    qC,qS,qD = align_edges(pC,pS,qC,qS,qD)
                    cnr = I.mesh.connectivity.face_to_first_corner(T2)
                    I.UVs[cnr + iCT2] = qC.xy
                    I.UVs[cnr + iDT2] = qD.xy
                    I.UVs[cnr + iST2] = qS.xy
                    T2n, iCT2, iDT2 = I.mesh.half_edges.opposite(C,D,T2)
                    push(T2, T2n, C,iCT2, qC, D, iDT2, qD)

                else : # build T1 in one side and T2 in another
                    T1, iAT1, iST1 = I.mesh.half_edges.opposite(A,S,T) # T1 on side of A
                    if T1 is not None and can_traverse(A,S):
                        visited[T1] = True
                        iCT1 = 3 - iAT1 - iST1
                        # build C from known points
                        C = I.mesh.ith_vertex_of_face(T1, iCT1)
                        S = I.mesh.ith_vertex_of_face(T1, iST1)
                        qA,qC,qS = build_triangle_singular(T1, iAT1, iCT1)
                        qA,qS,qC = align_edges(pA,pS,qA,qS,qC)
                        cnr = I.mesh.connectivity.face_to_first_corner(T1)
                        I.UVs[cnr + iAT1] = qA.xy
                        I.UVs[cnr + iCT1] = qC.xy
                        I.UVs[cnr + iST1] = qS.xy
                        T1n, iAT1, iCT1 = I.mesh.half_edges.opposite(A,C,T1)
                        push(T1,T1n, A, iAT1, qA, C, iCT1, qC)

                    T2, iBT2, iST2 = I.mesh.half_edges.opposite(B,S,T) # T2 on side of B
                    if T2 is not None and can_traverse(B,S):
                        visited[T2] = True
                        iDT2 = 3 - iBT2 - iST2
                        D = I.mesh.ith_vertex_of_face(T2, iDT2) 
                        S = I.mesh.ith_vertex_of_face(T2, iST2)
                        qB,qD,qS = build_triangle_singular(T2, iBT2, iDT2)
                        qB,qS,qD = align_edges(pB,pS,qB,qS,qD)
                        cnr = I.mesh.connectivity.face_to_first_corner(T2)
                        I.UVs[cnr + iBT2] = qB.xy
                        I.UVs[cnr + iDT2] = qD.xy
                        I.UVs[cnr + iST2] = qS.xy
                        T2n, iBT2, iDT2 = I.mesh.half_edges.opposite(B,D,T2)
                        push(T2,T2n, B, iBT2, qB, D, iDT2, qD)
            
            else: # regular triangle
                iCT = 3 - iAT - iBT
                C = I.mesh.faces[T][iCT]
                qA,qB,qC = [ build_triangle(T)[_x] for _x in (iAT, iBT, iCT)]
                qA,qB,qC = align_edges(pA,pB,qA,qB,qC)
                cnr = I.mesh.connectivity.face_to_first_corner(T)
                I.UVs[cnr + iAT] = qA.xy
                I.UVs[cnr + iBT] = qB.xy
                I.UVs[cnr + iCT] = qC.xy

                T1, iAT, iCT = I.mesh.half_edges.opposite(A,C,T)
                push(T, T1, A, iAT, qA, C, iCT, qC)

                T2, iBT, iCT = I.mesh.half_edges.opposite(B,C,T)
                push(T, T2, B, iBT, qB, C, iCT, qC)
        self.log("Tree traversal done.")

        self.log("Scaling and alignement with axes")
        I.UVs = rescale_uvs(I)
        
        self.log("Reposition singularities ")
        # n_singus_fail = 0 
        # n_singus_fail = replace_singularities_conformal_and_auth(I)
        n_singus_fail = replace_singularities_barycenter(I)
        # n_singus_fail = replace_singularities_ARAP_Jac(I)
        
        if n_singus_fail>0:
            self.log(f"/!\ {n_singus_fail} singularities failed to be positionned inside their triangle")
        else:
            self.log("All singularities have been positionned")


        self.reconstructed = True
        self.instance.cut_graph = self.cutter.cut_graph
