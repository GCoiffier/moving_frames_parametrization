import mouette as M
import sys
from os.path import join

"""
visualize.py

This scripts takes as input an output of the algorithm, and exports seams and singularities as obj files for vizualisation and render.

Usage:
    ` python visualize.py <folder_1> ... <folder_n>

where input folders are folders inside `output/` containing the .obj
"""

def import_custom_fields(path :str):
    with open(path, 'r') as objf:
        lines = objf.readlines()
    singu = []
    seams = []
    feat = []
    for line in lines:
        if "#" in line : continue
        toks = line.split()
        if not toks : continue
        identifier = toks[0]
        if identifier == 'c':
            # cone format: c <vertex-index> <k>
            vid = int(toks[1])-1
            cone = float(toks[2])
            singu.append( (vid, cone) )
        elif identifier == 'sm':
            # Seam format: sm <vertex-index-1> <vertex-index-1>
            v1 = int(toks[1])-1
            v2 = int(toks[2])-1
            seams.append((v1,v2))
        elif identifier == 'ft':
            # Seam format: ft <vertex-index-1> <vertex-index-1>
            v1 = int(toks[1])-1
            v2 = int(toks[2])-1
            feat.append((v1,v2))
    return singu, seams, feat

def get_singu_point_clouds(singu, mesh):
    PCneg = M.mesh.new_point_cloud()
    PCpos = M.mesh.new_point_cloud()
    PCcorner = M.mesh.new_point_cloud()
    for vid,c in singu:
        pt = mesh.vertices[vid]
        if c==1:
            PCpos.vertices.append(pt)
        elif c==-1:
            PCneg.vertices.append(pt)
        else:
            PCcorner.vertices.append(pt)
    return PCpos, PCneg, PCcorner

def get_seams(seams, mesh : M.mesh.SurfaceMesh) -> M.mesh.PolyLine:
    seamMesh = M.mesh.new_polyline()
    hard_edges = seamMesh.edges.create_attribute("hard_edges",bool) # otherwise edges are ignored
    new_v_id = dict()
    vid = 0
    for v1,v2 in seams:
        for v in (v1,v2):
            if v not in new_v_id:
                new_v_id[v] = vid
                vid += 1
                seamMesh.vertices.append(mesh.vertices[v])
        e = M.utils.keyify(new_v_id[v1], new_v_id[v2])
        hard_edges[len(seamMesh.edges)] = True
        seamMesh.edges.append(e)
    return seamMesh

def generate_visualization(mesh_name, subfolder=None):
    if subfolder is None:
        mesh_path = f"output/{mesh_name}/"
    else:
        mesh_path = subfolder
    mesh_file = join(mesh_path, f"{mesh_name}.obj")
    mesh = M.mesh.load(mesh_file)

    singus, seams, feat = import_custom_fields(mesh_file) # parse additionnal data
    
    seamsMesh = get_seams(seams,mesh)
    #seamsMesh = M.procedural.cylindrify_edges(seamsMesh, N=30, radius=0.01)
    M.mesh.save(seamsMesh, join(mesh_path, "visu_seams.obj"))

    featMesh = get_seams(feat, mesh)
    #featMesh = M.procedural.cylindrify_edges(featMesh, N=30, radius=0.01)
    M.mesh.save(featMesh,  join(mesh_path, "visu_feats.obj"))

    singuPos, singuNeg, singuCorner = get_singu_point_clouds(singus, mesh)
    
    for PC,which in ((singuPos, "pos"), (singuNeg, "neg"), (singuCorner,"corner")):
        if len(PC.vertices)==0 : continue
        #PC = M.procedural.spherify_vertices(PC, radius=1e-4)
        M.mesh.save(PC, join(mesh_path, f"visu_singu_{which}.obj"))

if __name__=="__main__":
    input_folders = sys.argv[1:]

    for folder in input_folders:
        print(folder)
        generate_visualization(folder)