:: to execute this batch file, cd to project directory and type "runs\runs.bat"

@REM python main.py val_per2_17
@REM python main.py val_per2_18
@REM python main.py val_per2_19

@REM python main_results.py val_per2_17 all 1300 1400
@REM python main_results.py val_per2_18 all 1300 1400
@REM python main_results.py val_per2_19 all 1300 1400

python main.py val_per2_18
python main_results.py val_per2_18 all 1300 1400