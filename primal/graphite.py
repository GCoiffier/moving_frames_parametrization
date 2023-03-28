import os

def generate_graphite_lua(mesh_name):
    folder = os.getcwd()
    folder = os.path.join(folder, f"output/{mesh_name}")
    text = "\
-- Lua (Keep this comment, this is an indication for editor's 'run' command) \n\
\n\
scene_graph.clear() \n\
main.camera().set_auto_focus(false)\n"

    # Initial frame field
    text+=f"\
ff = scene_graph.load_object(\"{folder}/framefield_init.mesh\")\n\
if (ff ~=nil)\n\
then\n\
	ff.shader.mesh_style='off; 0 0 0 1;1'\n\
	ff.shader.edges_style='true; 1 0 0 1;3'\n\
	ff.visible = false\n\
end\n\n"

    # Final frame field
    text+=f"\
ff = scene_graph.load_object(\"{folder}/framefield_final.mesh\")\n\
if (ff ~=nil)\n\
then\n\
	ff.shader.mesh_style='off; 0 0 0 1;1'\n\
	ff.shader.edges_style='true; 0 0 1 1;3'\n\
	ff.visible = false\n\
end\n\n"

    # Mesh with texture
    text+=f"\
out = scene_graph.load_object(\"{folder}/{mesh_name}.obj\")\n\
if (out ~= nil)\n\
then\n\
	out.visible = true\n\
	out.shader.painting = 'TEXTURE'\n\
	out.shader.tex_coords = 'facet_corners.tex_coord'\n\
	out.shader.tex_repeat = 10\n\
	out.shader.mesh_style = \"true; 0 0 0 1;0\"\n\
    out.shader.edges_style = \"off; 0.2 0.2 0.2 1;1\"\n\
end\n\n"
    
    # Cuts as red edges
    text+=f"\
cut = scene_graph.load_object(\"{folder}/seams.mesh\")\n\
if (cut ~=nil)\n\
then\n\
	cut.shader.mesh_style='off; 0 0 0 1;1'\n\
	cut.shader.edges_style=\"true; 1 0 0 1;5\"\n\
	cut.shader.vert_select_style=\"off; 1 0 0; 1\"\n\
	cut.visible = true\n\
end\n\n"

    # Seams as red edges
    text+=f"\
feat = scene_graph.load_object(\"{folder}/features.mesh\")\n\
if (feat ~=nil)\n\
then\n\
	feat.shader.mesh_style='false; 0 0 0 1;1'\n\
	feat.shader.edges_style=\"true; 1 0 0 1;5\"\n\
	feat.shader.vert_select_style=\"off; 1 0 0; 1\"\n\
	feat.visible = true\n\
end\n\n"

    # Singularity point cloud
    text+=f"\
singu = scene_graph.load_object(\"{folder}/singularities.geogram_ascii\")\n\
if (singu ~=nil)\n\
then\n\
	singu.visible = true\n\
	singu.shader.vertices_style = 'true; 0 0 0 0; 3'\n\
	singu.shader.painting = 'ATTRIBUTE'\n\
	singu.shader.attribute = 'vertices.index'\n\
	singu.shader.attribute_min = -2\n\
	singu.shader.attribute_max = 2\n\
end\n\n"

    # Flat Param
    text+=f"\
flat = scene_graph.load_object(\"{folder}/param_flat.geogram_ascii\")\n\
if (flat ~= nil)\n\
then \n\
	flat.visible = false\n\
	flat.shader.mesh_style='true; 0 0 0 1; 1'\n\
end\n"

    with open(f"output/{mesh_name}/show_in_graphite.lua", 'w') as f:
        f.write(text)