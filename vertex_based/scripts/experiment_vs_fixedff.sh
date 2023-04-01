python3 main.py -debug-output -visu-output $1 -init-mode smooth -optim-fixed-ff -n 500 -o experiment_fixedFF
python3 main.py -debug-output -visu-output $1 -dist lscm -n 500 -o experiment_lscm
python3 main.py -debug-output -visu-output $1 -dist iso -n 500 -o experiment_arap
python3 main.py -debug-output -visu-output $1 -dist area -n 500 -o experiment_arap

