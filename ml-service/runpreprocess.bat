@echo off
echo Running data preprocessing...
setlocal
set PYTHONPATH=%CD%

REM optional: generate synthetic CSV if missing
if not exist data\potato_water_requirement.csv (
    echo Generating synthetic dataset...
    python data\generate_synthetic.py
) else (
    echo Found existing data\potato_water_requirement.csv
)

python preprocess\fetch.py
python preprocess\preprocessing.py
python preprocess\features.py

endlocal
echo Preprocessing completed.
pause
