# Seamless Global Parametrization in a single optimization.

Link to Paper : <todo>

### How to use

First install the required python packages by running
`pip3 install requirements.txt`

Then run with the command 
`python3 main.py path/to/input/mesh`

Available input mesh formats : .obj, .mesh (.medit), .geogram_ascii

optional arguments:
  -h, --help            show this help message and exit

  -outname OUTPUT_NAME, --output-name OUTPUT_NAME
                        Name of the output file. Will be stored in a folder
                        named '<outname>' as '<outname>.obj'

  -n N_ITER_MAX, --n-iter-max N_ITER_MAX
                        maximum number of iterations in optimization

  -dist {None,LSCM,shear,iso}, --distortion {None,LSCM,shear,iso}
                        choice of distortion energy
                        
  -fixed-ff, --fixed-ff
                        Runs the optimization with a fixed pre-computed frame
                        field.

  -feat, --detect-features
                        enables feature detection and alignment

  -no-tqdm, --no-tqdm   disables tqdm progress bar

  -debug-output, --debug-output
                        Debug output. This options outputs various meshes on
                        top of the standard .obj output

  -visu-output, --visu-output
                        Visualization output. This options outputs
                        singularities, seams and features as surface meshes
                        for rendering and visualization.
