#python3 main.py -debug-output -visu-output ~/mesh/surface/tree.obj -init-mode smooth -optim-fixed-ff -n 500 -o tree_fixedFF
#python3 main.py -debug-output -visu-output ~/mesh/surface/tree.obj -dist lscm -n 500 -o tree_lscm
#python3 main.py -debug-output -visu-output ~/mesh/surface/tree.obj -dist iso -n 500 -o tree_arap

python3 main.py -debug-output -visu-output ~/mesh/surface/rocker_arm.obj -init-mode smooth -optim-fixed-ff -n 500 -o rocker_arm_fixedFF
python3 main.py -debug-output -visu-output ~/mesh/surface/rocker_arm.obj -init-mode smooth -dist lscm -n 500 -o rocker_arm_lscm
python3 main.py -debug-output -visu-output ~/mesh/surface/rocker_arm.obj -init-mode smooth -dist iso -n 500 -o rocker_arm_arap
