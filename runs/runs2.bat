:: to execute this batch file, cd to project directory and type "runs\runs.bat"

python main.py val_per2_17
python main.py val_per2_18
python main.py val_per2_19

python main_results.py val_per2_17 all 1300 1400
python main_results.py val_per2_18 all 1300 1400
python main_results.py val_per2_19 all 1300 1400

