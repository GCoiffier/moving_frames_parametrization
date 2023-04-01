python3 main.py -debug-output -visu-output $1 -init-mode smooth -order 4  -o experiment_FF4
python3 main.py -debug-output -visu-output $1 -init-mode smooth -order 6  -o experiment_FF6
python3 main.py -debug-output -visu-output $1 -init-mode smooth -order 10 -o experiment_FF10
python3 main.py -debug-output -visu-output $1 -init-mode smooth -order 20 -o experiment_FF20
