python3 main.py -debug-output -visu-output $1 -n 1000 -init-mode smooth -dist none -o experiment_fixedff
python3 main.py -debug-output -visu-output $1 -n 1000 -dist none -o experiment_nodist
python3 main.py -debug-output -visu-output $1 -n 500  -dist iso -o experiment_iso
python3 main.py -debug-output -visu-output $1 -n 500  -dist lscm -o experiment_lscm
python3 main.py -debug-output -visu-output $1 -n 500  -dist area -o experiment_area