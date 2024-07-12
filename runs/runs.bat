:: to execute this batch file, cd to project directory and type "runs\runs.bat"

python main.py val3_xb
python main.py val4
python main.py val4_xb

python main_results.py val3_xb bluff_and_toe
python main_results.py val3_xb bed
python main_results.py val3_xb heat
python main_results.py val3_xb temp_heat
python main_results.py val3_xb temp

python main_results.py val4 bluff_and_toe
python main_results.py val4 bed
python main_results.py val4 heat
python main_results.py val4 temp_heat
python main_results.py val4 temp

python main_results.py val4_xb bluff_and_toe
python main_results.py val4_xb bed
python main_results.py val4_xb heat
python main_results.py val4_xb temp_heat
python main_results.py val4_xb temp