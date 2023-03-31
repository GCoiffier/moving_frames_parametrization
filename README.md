# Moving Frames Surface Parametrization

![](https://repository-images.githubusercontent.com/620222816/7d620031-9c67-4ed1-864d-3c72dcd8ad14)

Global seamless parametrization algorithm for triangular meshes using Cartan's method of moving frames

## Installation and dependencies

From the main folder, run the command:
```
pip install -r requirements.txt
```

This will install the needed python modules and their dependencies :
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/) (support for sparse matrices)
- [tqdm](https://tqdm.github.io/) (neat formatting for logs)
- [numba](https://numba.pydata.org/) (jit compilation and energy computation speedup)
- [osqp](https://osqp.org/) (quadratic programming solver)
- [mouette](https://github.com/GCoiffier/mouette), our library for the handling of mesh data structures as well as classical geometry processing algorithms

Additionnally, we rely on Intel `oneMKL pardiso` solver for the internal solver of OSQP (https://osqp.org/docs/get_started/linear_system_solvers.html). To install, follow the instructions on their website : https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
If the installation is a problem, you can switch to OSQP's default solver `qdldl` by manually setting the variable at the top of `src/optimize.py`

The code has been tested on Ubuntu 18.04, 20.04 and 22.04.

## The Dual algorithm

This version of the algorithm is the code accompanying the paper and used to generate all the results.

#### Usage
```
python main.py [-h] [-o OUTPUT_NAME] [-n N_ITER_MAX] [-dist {none,lscm,arap,area}] [-fbnd] [-init-mode {zero,smooth,curv,random}] [-optim-fixed-ff] [-order ORDER] [-feat]
               [-no-tqdm] [-silent] [-debug-output] [-visu-output]
               input_mesh

positional arguments:
  input_mesh
        path to the input mesh. Supported formats: .obj, .mesh, .geogram_ascii

options:
  -h, --help
        show this help message and exit

  -o OUTPUT_NAME, --output-name OUTPUT_NAME
        Name of the output file. Will be stored in a folder named '<outname>' as '<outname>.obj'

  -n N_ITER_MAX, --n-iter-max N_ITER_MAX
        maximum number of iterations in optimization

  -dist {none,lscm,arap,area}, --distortion {none,lscm,arap,area}
        choice of distortion energy

  -fbnd, --free-boundary
        Free boundary - No cones mode

  -init-mode {zero,smooth,curv,random}, --init-mode {zero,smooth,curv,random}
        Initialization mode for frame field and rotations

  -optim-fixed-ff, --optim-fixed-ff
        Runs the optimization with a fixed pre-computed frame field.

  -order ORDER, --order ORDER
        order of the frame field

  -feat, --detect-features
        enables feature detection and alignment

  -no-tqdm, --no-tqdm
        disables tqdm progress bar

  -silent, --silent
        disables output in terminal

  -debug-output, --debug-output
        Debug output. This options outputs various meshes on top of the standard .obj output

  -visu-output, --visu-output
        Visualization output. This options outputs singularities, seams and features as .obj files for rendering and visualization.
```

## The Primal algorithm

The primal version is another discretization using mesh triangles instead of vertex charts. This leads to less variables in the optimization but the algorithm is no longer provably correct (nothing prevents double coverings from appearing, though this does not seem to happen in practice)

#### Usage

```
python main.py [-h] [-o OUTPUT_NAME] [-n N_ITER_MAX] [-dist {none,lscm,lscm_metric,arap,arap_metric,id,id_cst,id_metric,area,area_metric}] [-init-smooth] [-optim-fixed-ff]
            [-feat] [-no-tqdm] [-silent] [-debug-output] [-visu-output]
            input_mesh

positional arguments:
  input_mesh            path to the input mesh. Supported formats: .obj, .mesh, .geogram_ascii

options:
  -h, --help
        show this help message and exit

  -o OUTPUT_NAME, --output-name OUTPUT_NAME
        Name of the output file. Will be stored in a folder named '<outname>' as '<outname>.obj'

  -n N_ITER_MAX, --n-iter-max N_ITER_MAX
        maximum number of iterations in optimization

  -dist {none,lscm,lscm_metric,arap,arap_metric,id,id_cst,id_metric,area,area_metric}, --distortion {none,lscm,lscm_metric,arap,arap_metric,id,id_cst,id_metric,area,area_metric}
        choice of distortion

  -init-smooth, --init-smooth
        Initializes the frame field as a smooth one (vs zeros everywhere)

  -optim-fixed-ff, --optim-fixed-ff
        Runs the optimization with a fixed frame field.

  -feat, --detect-features
        enables feature detection and alignment

  -no-tqdm, --no-tqdm   
        disables tqdm progress bar

  -silent, --silent
        disables output in terminal

  -debug-output, --debug-output
        Debug output. This options outputs various meshes on top of the standard .obj output

  -visu-output, --visu-output
        Visualization output. This options outputs singularities, seams and features as surface meshes for rendering and visualization.
```

#### Initialization options
Unlike the dual version, this version only provides initialization `zero` and `smooth` for the frame field.


#### Distortion options
Additionnal experimental distortions have been implemented in this version. Along the classical distortion energy (similar to their dual counterpart), it is possible :
- to penalize jacobian that are not the identity matrix (`id`)
- to alter the metric inside the Levenberg-Marquardt descent direction, thus penalizing parametrizations that are not conformal (distortion `lscm_metric`), not isometric (`arap_metric`), not authalic (`area_metric`) or not the identity (`id_metric`).
- to enforce the distortion as a feasible region that will grow over time, using linear constraints in the optimization. This is done for the identity distortion (`id_cst`).