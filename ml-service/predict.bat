@echo off
REM Run irrigation predictor with CLI values

REM ---- EDIT THESE IF NEEDED ----
SET MODEL_PATH=models\soil_moisture_pump_model.pkl
SET SCRIPT_DIR=%~dp0
REM --------------------------------

IF "%~1"=="" (
    echo Usage: predictA.bat SOIL TEMP HUM
    echo Example: predictA.bat 45 28 55
    pause
    exit /b
)

SET SOIL=%1
SET TEMP=%2
SET HUM=%3

python "%SCRIPT_DIR%src\predict.py" ^
    --model "%SCRIPT_DIR%%MODEL_PATH%" ^
    --soil %SOIL% ^
    --temp %TEMP% ^
    --hum %HUM%

pause
