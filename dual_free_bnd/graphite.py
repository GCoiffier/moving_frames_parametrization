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
    
    # Flat Param
    text+=f"\
flat = scene_graph.load_object(\"{folder}/param_flat.obj\")\n\
if (flat ~= nil)\n\
then \n\
	flat.visible = false\n\
	flat.shader.mesh_style='true; 0 0 0 1; 1'\n\
end\n"

    with open(f"output/{mesh_name}/show_in_graphite.lua", 'w') as f:
        f.write(text)