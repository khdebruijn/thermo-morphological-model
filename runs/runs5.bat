:: to execute this batch file, cd to project directory and type "runs\results.bat"

@REM python main.py val_per2_4
@REM python main_results.py val_per2_4 all 1300 1400

python main.py val_per1_base
python main.py val_per1_no-therm

python main.py val_per2_base
python main.py val_per2_no-therm

python main.py val_per3_base
python main.py val_per3_no-therm
