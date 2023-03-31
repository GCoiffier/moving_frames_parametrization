from src.common import Distortion, Options, VerboseOptions
from src.instance import Instance
from src.initialize import Initializer
from src.optimize import Optimizer
from src.reconstruct import ParamConstructor, write_output_obj
from visualize import generate_visualization
from graphite import generate_graphite_lua

import mouette as M
import argparse
import numpy as np
import os
import csv
from time import time

def export_dict_as_csv(data : dict, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

np.set_printoptions(threshold=1000000, precision=3, linewidth=np.nan)

DIST_CHOICES = [
    "none",         # No distortion
    "lscm",         # Conformal distortion by energy
    "lscm_metric",  # Conformal distortion by change of metric in optimizer
    "arap",         # isometric distortion by energy
    "arap_metric",  # isometric distortion by change of metric
    "id",           # identity distortion by energy
    "id_cst",       # identity distortion by linear constraints
    "id_metric",    # identity distortion by change of metric
    "area",         # area distortion by energy
    "area_metric",  # area distortion by change of metric
]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh. Supported formats: .obj, .mesh, .geogram_ascii")

    parser.add_argument("-o", "--output-name", type=str, default="", \
        help="Name of the output file. Will be stored in a folder named '<outname>' as '<outname>.obj'")

    parser.add_argument("-n", "--n-iter-max", type=int, default=500, \
        help="maximum number of iterations in optimization")

    parser.add_argument("-dist", "--distortion", type=str, choices=DIST_CHOICES, default="none", \
        help="choice of distortion energy")

    parser.add_argument("-init-smooth", "--init-smooth", action="store_true", \
        help="Initializes the frame field as a smooth one (vs zeros everywhere)")

    parser.add_argument("-optim-fixed-ff", "--optim-fixed-ff", action="store_true", \
        help="Runs the optimization with a fixed frame field.")

    parser.add_argument("-feat", "--detect-features", action="store_true", \
        help="enables feature detection and alignment")
    
    parser.add_argument("-no-tqdm", "--no-tqdm", action="store_true",\
        help="disables tqdm progress bar")

    parser.add_argument("-silent", "--silent", action="store_true",\
        help="disables output in terminal")

    parser.add_argument("-debug-output", "--debug-output", action="store_true",
        help="Debug output. This options outputs various meshes on top of the standard .obj output")

    parser.add_argument("-visu-output", "--visu-output", action="store_true",
        help="Visualization output. This options outputs singularities, seams and features as surface meshes for rendering and visualization.")

    args = parser.parse_args()

    if len(args.output_name)==0:
        args.output_name = M.utils.get_filename(args.input_mesh)

    ###### Load data ###### 
    mesh = M.mesh.load(args.input_mesh)
    FOLDER = os.path.join("output", args.output_name)
    MAIN_OBJ_OUTPUT = f"output/{args.output_name}/{args.output_name}.obj"
    os.makedirs(FOLDER, exist_ok=True)

    ###### Initialization ######
    START_TIME = time()

    verbose_options = VerboseOptions(
        output_dir=FOLDER,
        logger_verbose= not args.silent,
        qp_solver_verbose=False,
        optim_verbose= not args.silent,
        tqdm= not args.no_tqdm,
        log_freq=1
    )

    options = Options(
        initSmooth=args.init_smooth,
        optimFixedFF=args.optim_fixed_ff,
        distortion=Distortion.from_string(args.distortion),
        features= args.detect_features,
        n_iter_max=args.n_iter_max
    )
    options.set_schedule()
    
    if not args.silent: print(options)

    instance = Instance(mesh)
    init = Initializer(instance, options=options, verbose_options=verbose_options)
    init.initialize()
    reconstructor = ParamConstructor(instance, options=options, verbose_options=verbose_options)

    if args.debug_output:
        init_ff = reconstructor.export_frame_field()
        M.mesh.save(init_ff, os.path.join(FOLDER, "framefield_init.mesh")) # save as .mesh file

        feature_graph = reconstructor.export_feature_graph()
        M.mesh.save(feature_graph, os.path.join(FOLDER,"features.mesh"))
    
    ##### Optimization #####
    optim = Optimizer(instance, options=options, verbose_options=verbose_options)
    final_energy = optim.optimize()
    
    ##### Reconstruction and export #####
    reconstructor.construct_param()
    write_output_obj(instance, MAIN_OBJ_OUTPUT)
    END_TIME = time()

    # Distortion
    distMeasure = M.processing.parametrization.ParamDistortion(instance.mesh)()
    report = distMeasure.summary
    report["time"] = END_TIME - START_TIME
    report["energy"] = final_energy
    export_dict_as_csv(report, os.path.join(FOLDER,"summary.csv"))

    n_singus = 0
    for v in instance.singular_vertices:
        if round(instance.singular_vertices[v]) != 0: n_singus += 1
    
    if not args.silent:
        print("Distortion:\n", report)
        print("Singularities :", n_singus)

    if args.debug_output:
        # additionnal outputs
        M.mesh.save(instance.mesh, os.path.join(FOLDER,"uvs.geogram_ascii"))
        final_ff = reconstructor.export_frame_field()
        M.mesh.save(final_ff, os.path.join(FOLDER,"framefield_final.mesh"))

        seam_graph = reconstructor.export_seams()
        M.mesh.save(seam_graph, os.path.join(FOLDER,"seams.mesh"))

        flat_param = reconstructor.export_flat_mesh()
        M.mesh.save(instance.flat_mesh, os.path.join(FOLDER,"param_flat.geogram_ascii")) # for attributes
        M.mesh.save(instance.flat_mesh, os.path.join(FOLDER,"param_flat.obj")) # for compatibility

        disk_mesh = reconstructor.export_disk_mesh()
        M.mesh.save(disk_mesh, os.path.join(FOLDER,"uvs_disk.obj"))
        
        singuls = reconstructor.export_singularity_point_cloud()
        M.mesh.save(singuls, os.path.join(FOLDER, "singularities.geogram_ascii")) # for attributes
        M.mesh.save(singuls, os.path.join(FOLDER, "singularities.xyz")) # for compatibility

        M.mesh.save(instance.mesh, os.path.join(FOLDER,"work_mesh.geogram_ascii"))

        generate_graphite_lua(args.output_name)
    
    if args.visu_output:
        generate_visualization(args.output_name)