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
    ref_frame = complex(I.var[I.var_sep_ff], I.var[I.var_sep_ff+1])
    angle = cmath.phase(ref_frame)

    # apply transformation
    for c in I.mesh.id_corners:
        I.UVs[c] = rotate_2d(I.UVs[c] / scale, angle)
    return I.UVs

def write_output_obj(I : Instance, file_path : str):
    """Final export of the mesh as an obj file with custom fields for singularity cones, seams and feature edges"""
    M.mesh.save(I.mesh, file_path)

class ParamConstructor(Worker):
    """Worker responsible for putting the parametrization back together after optimization. Also exports various debug outputs"""

    def __init__(self, instance: Instance, options = Options(), verbose_options = VerboseOptions()):
        super().__init__("ParamReconstruction", instance, options, verbose_options)
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
        return I.param_mesh

    def construct_param(self):
        I = self.instance
        self.log("Starting UV reconstruction.")
        # We reconstruct along a spanning tree whose root is not singular
        root = 0
        visited = M.ArrayAttribute(bool, len(I.mesh.faces)) #  I.mesh.faces.create_attribute("visited", bool, dense=True)
        I.UVs = I.mesh.face_corners.create_attribute("uv_coords", float, 2)
        queue = deque()
        I.barycenters = M.attributes.face_barycenter(I.mesh, persistent=False) # recompute barycenters since we have split some triangles

        def build_edge(A,B):
            ie = I.mesh.connectivity.edge_id(A,B)
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

        def build_triangle(T):
            A,B,C = I.mesh.faces[T]
            pA = M.Vec.zeros(3)
            pB = build_edge(A,B)
            pC = build_edge(A,C)
            return pA,pB,pC

        def push(T, T2, A, iAT, pA, B, iBT, pB):
            if (T2 is not None) and (not visited[T2]):
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
        self.reconstructed = True
