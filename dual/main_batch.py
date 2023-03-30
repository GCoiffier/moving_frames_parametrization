"""
main_batch.py
Runs the parametrization algorithm on all the meshes found in the given folder, with the same parameters on all models
"""

import os
import csv
import argparse
import numpy as np
from time import time

import mouette as M
from src.common import export_dict_as_csv, Distortion, InitMode, Options, VerboseOptions
from src.instance import Instance
from src.initialize import Initializer
from src.optimize import Optimizer
from src.reconstruct import ParamConstructor, write_output_obj

from visualize import generate_visualization

np.set_printoptions(threshold=1000000, precision=3, linewidth=np.nan)

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="path to the input mesh")
    parser.add_argument("-report-name", "--report-name", default="", help="name of the final csv report")
    
    parser.add_argument("-dist", "--distortion", type=str, choices=["none", "lscm", "conf", "arap", "iso", "area", "scale"], default="none", \
        help="choice of distortion energy")
    
    parser.add_argument("-init-mode", "--init-mode", type=str, choices=["zero", "smooth", "curv", "random"], default="zero", \
        help="Initialization mode for frame field and rotations")

    parser.add_argument("-optim-fixed-ff", "--optim-fixed-ff", action="store_true", \
        help="Runs the optimization with a fixed pre-computed frame field.")
   
    parser.add_argument("-feat", "--detect-features", action="store_true", \
        help="enables feature detection and alignment")
    
    parser.add_argument("-resume", "--resume", action="store_true", help="skips input models for which an output already exists in order to resume the run")

    args = parser.parse_args()
    
    all_files = os.listdir(args.dir)
    all_files.sort()
    print(all_files)
    
    with open(f"output/report{args.report_name}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["model", "energy", "time", "n_singus", "triangles", "conformal", "authalic", "det", "log_det", "iso", "shear", "stretch_max", "stretch_mean"])
        writer.writeheader()
        
        for inputmesh in all_files:
            
            print(inputmesh)
            try:

                ###### Load data ###### 
                output_name = M.utils.get_filename(inputmesh)
                input_path = os.path.join(args.dir, inputmesh)
                mesh = M.mesh.load(input_path)
                FOLDER = os.path.join("output", output_name)
                MAIN_OBJ_OUTPUT = f"output/{output_name}/{output_name}.obj"
                if os.path.exists(FOLDER) and args.resume : continue
                os.makedirs(FOLDER, exist_ok=True)

                ###### Initialization ###### 
                START_TIME = time()

                verbose = VerboseOptions(
                    output_dir = FOLDER,
                    logger_verbose=False,
                    qp_solver_verbose=False,
                    optim_verbose=False,
                    tqdm= False,
                    log_freq=1
                )

                options = Options(
                        distortion = Distortion.from_string(args.distortion),
                        features = args.detect_features,
                        initMode = InitMode.from_string(args.init_mode),
                        optimFixedFF = args.optim_fixed_ff,
                        n_iter_max = 300
                    )
                options.set_schedule()

                instance = Instance(mesh)
                init = Initializer(instance, options=options, verbose_options=verbose)
                init.initialize()
                reconstructor = ParamConstructor(instance, options=options, verbose_options=verbose)
                init_ff = reconstructor.export_frame_field()
                M.mesh.save(init_ff, os.path.join(FOLDER, "framefield_init.mesh")) # save as .mesh file

                feature_graph = reconstructor.export_feature_graph()
                M.mesh.save(feature_graph, os.path.join(FOLDER,"features.mesh"))

                ##### Optimization #####
                optim = Optimizer(instance, options=options, verbose_options=verbose)
                final_energy = optim.optimize()
                
                ##### Reconstruction and export #####
                reconstructor.construct_param()
                write_output_obj(instance, MAIN_OBJ_OUTPUT)
                END_TIME = time()

                # Distortion
                distMeasure = M.processing.parametrization.ParamDistortion(instance.mesh)()
                report = distMeasure.summary
                print(distMeasure.summary)

                report["time"] = END_TIME - START_TIME
                report["triangles"] = len(instance.mesh.faces)
                export_dict_as_csv(report, os.path.join(FOLDER,"summary.csv"))
                report["model"] = output_name
                report["energy"] = final_energy
                report["n_singus"] = len(instance.singular_vertices)

                M.mesh.save(instance.mesh, os.path.join(FOLDER,"uvs.geogram_ascii"))
                final_ff = reconstructor.export_frame_field()
                M.mesh.save(final_ff, os.path.join(FOLDER,"framefield_final.mesh"))

                seam_graph = reconstructor.export_seams()
                M.mesh.save(seam_graph, os.path.join(FOLDER,"seams.mesh"))

                flat_param = reconstructor.export_flat_mesh()
                M.mesh.save(instance.param_mesh, os.path.join(FOLDER,"param_flat.geogram_ascii")) # for attributes

                disk_mesh = reconstructor.export_disk_mesh()
                M.mesh.save(disk_mesh, os.path.join(FOLDER,"uvs_disk.obj"))
                
                singuls = reconstructor.export_singularity_point_cloud()
                M.mesh.save(singuls, os.path.join(FOLDER, "singularities.geogram_ascii")) # for attributes

                M.mesh.save(instance.work_mesh, os.path.join(FOLDER,"work_mesh.geogram_ascii"))
                
                generate_visualization(output_name)
                writer.writerow(report)
            except Exception as e:
                print(e)
                print()
                continue