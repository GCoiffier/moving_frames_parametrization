#python3 main.py -debug-output -visu-output $1 -n 500 -dist none -init-mode smooth -o experiment_init_smooth
#python3 main.py -debug-output -visu-output $1 -n 500 -dist none -init-mode zero -o experiment_init_zero
#python3 main.py -debug-output -visu-output $1 -n 500 -dist none -init-mode curv -o experiment_init_curv
#python3 main.py -debug-output -visu-output $1 -n 500 -dist none -init-mode random -o experiment_init_random

#python3 main.py -debug-output -visu-output $1 -n 300 -dist lscm -init-mode smooth -o experiment_init_smooth_lscm
#python3 main.py -debug-output -visu-output $1 -n 300 -dist lscm -init-mode zero -o experiment_init_zero_lscm
#python3 main.py -debug-output -visu-output $1 -n 300 -dist lscm -init-mode curv -o experiment_init_curv_lscm
#python3 main.py -debug-output -visu-output $1 -n 300 -dist lscm -init-mode random -o experiment_init_random_lscm

#python3 main.py -debug-output -visu-output $1 -n 300 -dist iso -init-mode smooth -o experiment_init_smooth_arap
#python3 main.py -debug-output -visu-output $1 -n 300 -dist iso -init-mode zero -o experiment_init_zero_arap
#python3 main.py -debug-output -visu-output $1 -n 300 -dist iso -init-mode curv -o experiment_init_curv_arap
#python3 main.py -debug-output -visu-output $1 -n 300 -dist iso -init-mode random -o experiment_init_random_arap

python3 main.py -debug-output -visu-output $1 -n 100 -dist scale -init-mode smooth -o experiment_init_smooth_scale
python3 main.py -debug-output -visu-output $1 -n 100 -dist scale -init-mode zero -o experiment_init_zero_scale
python3 main.py -debug-output -visu-output $1 -n 100 -dist scale -init-mode curv -o experiment_init_curv_scale
python3 main.py -debug-output -visu-output $1 -n 100 -dist scale -init-mode random -o experiment_init_random_scale