@echo off
setlocal enabledelayedexpansion

:: === GET SCRIPT DIRECTORY ===
set "BASEDIR=%~dp0"
cd /d "%BASEDIR%"

:: === LOAD ENV VARIABLES FROM .env FILE ===
if exist .env (
    for /f "delims=" %%x in (.env) do set %%x
) else (
    echo .env file not found!
    exit /b
)

:: === CHECK IF PASSWORD IS LOADED ===
if "%MYSQL_PASSWORD%"=="" (
    echo MySQL password not set in .env file.
    exit /b
)

:: === DROP AND CREATE DATABASE ===
echo Dropping and creating database...
mysql -u root -p%MYSQL_PASSWORD% -e "DROP DATABASE IF EXISTS optuna_study;"
mysql -u root -p%MYSQL_PASSWORD% -e "CREATE DATABASE optuna_study;"

:: === CHECK IF ARGUMENT IS PROVIDED ===
if "%~1"=="" (
    echo Please provide the number of times to run the script.
    exit /b
)

set "runs=%~1"

:: === PREPARE PYTHON COMMAND ===
set "PYTHON_CMD=.venv\Scripts\activate && python -m src.main && pause"

:: === ATTACH TO EXISTING TERMINAL OR OPEN A NEW ONE ===
set "WT_COMMAND=wt -w 0 new-tab"
%WT_COMMAND% --title "Script 1" cmd /k "cd /d %BASEDIR% && %PYTHON_CMD%" || (
    echo No running Windows Terminal found. Starting a new one...
    start wt new-tab --title "Script 1" cmd /k "cd /d %BASEDIR% && %PYTHON_CMD%"
)

:: Add a 5-second delay before starting the rest
timeout /t 5 /nobreak >nul

:: === LOOP TO RUN REMAINING SCRIPTS ===
set /a count=1
:loop
if !count! lss %runs% (
    set /a count+=1
    echo Starting script !count!...
    %WT_COMMAND% --title "Script !count!" cmd /k "cd /d %BASEDIR% && %PYTHON_CMD%"
    timeout /t 5 /nobreak >nul
    goto loop
)

echo All scripts started!
