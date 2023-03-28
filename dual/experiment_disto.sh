python3 main.py -debug-output -visu-output ../../bench/rhino/rhino_20k.obj -n 1000 -init-mode smooth -dist none -o rhino_fixedff
python3 main.py -debug-output -visu-output ../../bench/rhino/rhino_20k.obj -n 1000 -dist none -o rhino_nodist
python3 main.py -debug-output -visu-output ../../bench/rhino/rhino_20k.obj -n 500 -dist iso -o rhino_iso
python3 main.py -debug-output -visu-output ../../bench/rhino/rhino_20k.obj -n 500 -dist lscm -o rhino_lscm
python3 main.py -debug-output -visu-output ../../bench/rhino/rhino_20k.obj -n 100 -dist id -o rhino_id
python3 main.py -debug-output -visu-output ../../bench/rhino/rhino_20k.obj -n 500 -dist shear -o rhino_shear