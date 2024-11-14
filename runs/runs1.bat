:: to execute this batch file, cd to project directory and type "runs\runs.bat"
:: (did not cancel yet, but can be cancelled)

python main.py val_per2_15
python main_results.py val_per2_15 all 1300 1400

python main.py val_per2_16
python main_results.py val_per2_16 all 1300 1400