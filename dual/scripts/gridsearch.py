"""
Gridsearch script for the parameter lambda_f (balance between (E) and (F) i.e. optimizer.FF_weight/optimizer.edge_weight)
Outputs results for a model using a range of values for this parameter
"""

import os
import csv
import argparse
import numpy as np
from time import time

import mouette as M
from src.common import export_dict_as_csv, Distortion, Options, VerboseOptions, InitMode
from src.instance import Instance
from src.initialize import Initializer
from src.optimize import Optimizer, OptimHyperParameters
from src.reconstruct import ParamConstructor, write_output_obj

from visualize import generate_visualization

np.set_printoptions(threshold=1000000, precision=3, linewidth=np.nan)

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh. Supported formats: .obj, .mesh, .geogram_ascii")

    parser.add_argument("-o", "--output-name", type=str, default="", \
        help="Name of the output file. Will be stored in a folder named '<outname>' as '<outname>.obj'")
    
    parser.add_argument("-dist", "--distortion", type=str, choices=["none", "lscm", "arap", "area"], default="none", \
        help="choice of distortion energy")
   
    parser.add_argument("-feat", "--detect-features", action="store_true", \
        help="enables feature detection and alignment")

    parser.add_argument("-init-mode", "--init-mode", type=str, choices=["zero", "smooth", "curv", "random"], default="zero", \
        help="Initialization mode for frame field and rotations")

    parser.add_argument("-optim-fixed-ff", "--optim-fixed-ff", action="store_true", \
        help="Runs the optimization with a fixed pre-computed frame field.")

    args = parser.parse_args()

    if len(args.output_name)==0:
        args.output_name = M.utils.get_filename(args.input_mesh)

    ###### Load data ###### 
    FOLDER = os.path.join("output", args.output_name)
    os.makedirs(FOLDER, exist_ok=True)

    ###### Initialization ###### 
    verbose = VerboseOptions(
        output_dir = FOLDER,
        logger_verbose=False,
        qp_solver_verbose=False,
        optim_verbose=False,
        tqdm=False,
        log_freq=1
    )

    options = Options(
        distortion = Distortion.from_string(args.distortion),
        features = args.detect_features,
        initMode = InitMode.from_string(args.init_mode),
        optimFixedFF = args.optim_fixed_ff,
        n_iter_max = 100
    )
    options.set_schedule()
    if options.distortion == Distortion.NONE:
        options.n_iter_max = 500

    options.set_schedule() # default schedule depending on distortion

    WEIGHTS = [(_w, 1.) for _w in np.logspace(1,2,15)][::-1] + [(1., _w) for _w in np.logspace(0, 3, 20)]
    with open( os.path.join(FOLDER, f"report.csv"), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Edge_weight", "FF_weight", "energy", "time", "n_singus", "conformal", "authalic", "det", "log_det", "iso", "shear", "stretch_max", "stretch_mean"])
        writer.writeheader()
        
        for i, (e_weight,ff_weight) in enumerate(WEIGHTS):
                print(e_weight, ff_weight)
                #try:
                ###### Initialization ###### 
                mesh = M.mesh.load(args.input_mesh)
                subfolder = os.path.join(FOLDER, str(i))
                os.makedirs(subfolder, exist_ok=True)
                print(subfolder, args.output_name)
            
                START_TIME = time()
                instance = Instance(mesh)
                init = Initializer(instance, options=options, verbose_options=verbose)
                init.initialize()
                reconstructor = ParamConstructor(instance, options=options, verbose_options=verbose)
                init_ff = reconstructor.export_frame_field()
                M.mesh.save(init_ff, os.path.join(subfolder, "framefield_init.mesh")) # save as .mesh file

                feature_graph = reconstructor.export_feature_graph()
                M.mesh.save(feature_graph, os.path.join(subfolder,"features.mesh"))

                ##### Optimization #####
                hp = OptimHyperParameters(
                    MIN_DELTA_E = 0, #1e-3,
                    MIN_STEP_NORM = 0, #1e-4,
                    MIN_GRAD_NORM = 0 #1e-6
                )
                optim = Optimizer(instance, options=options, verbose_options=verbose, optim_hp=hp)
                optim.edge_weight = e_weight
                optim.FF_weight = ff_weight
                final_energy = optim.optimize()
                
                ##### Reconstruction and export #####
                reconstructor.construct_param()
                write_output_obj(instance, os.path.join(subfolder, f"{args.output_name}.obj"))
                END_TIME = time()

                # Distortion
                distMeasure = M.processing.parametrization.ParamDistortion(instance.mesh)()
                report = distMeasure.summary
                report["Edge_weight"] = e_weight
                report["FF_weight"] = ff_weight
                report["n_singus"] = len(instance.singular_vertices)
                report["time"] = END_TIME - START_TIME
                report["energy"] = final_energy
                export_dict_as_csv(report, os.path.join(subfolder, "summary.csv"))

                M.mesh.save(instance.mesh, os.path.join(subfolder,"uvs.geogram_ascii"))
                final_ff = reconstructor.export_frame_field()
                M.mesh.save(final_ff, os.path.join(subfolder,"framefield_final.mesh"))

                seam_graph = reconstructor.export_seams()
                M.mesh.save(seam_graph, os.path.join(subfolder,"seams.mesh"))

                flat_param = reconstructor.export_flat_mesh()
                M.mesh.save(instance.param_mesh, os.path.join(subfolder,"param_flat.geogram_ascii")) # for attributes

                disk_mesh = reconstructor.export_disk_mesh()
                M.mesh.save(disk_mesh, os.path.join(subfolder,"uvs_disk.obj"))
                
                singuls = reconstructor.export_singularity_point_cloud()
                M.mesh.save(singuls, os.path.join(subfolder, "singularities.geogram_ascii")) # for attributes

                M.mesh.save(instance.work_mesh, os.path.join(subfolder,"work_mesh.geogram_ascii"))
                
                writer.writerow(report)

                generate_visualization(args.output_name, subfolder)

                # except Exception as e:
                #     print(e)
                #     print()
                #     continue