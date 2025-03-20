@echo off
setlocal enabledelayedexpansion

:: === LOAD ENV VARIABLES FROM .env FILE ===
if exist .env (
    for /f "delims=" %%x in (.env) do set %%x
) else (
    echo .env file not found!
    exit /b
)

:: === CHECK IF PASSWORD IS LOADED ===
if "%MYSQL_PWD%"=="" (
    echo MySQL password not set in .env file.
    exit /b
)

:: === DROP AND CREATE DATABASE ===
echo Dropping and creating database...
mysql -u root -p%MYSQL_PWD% -e "DROP DATABASE IF EXISTS my_database;"
mysql -u root -p%MYSQL_PWD% -e "CREATE DATABASE my_database;"

:: === CHECK IF ARGUMENT IS PROVIDED ===
if "%~1"=="" (
    echo Please provide the number of times to run the script.
    exit /b
)

set "runs=%~1"

:: === RUN FIRST SCRIPT IMMEDIATELY ===
echo Starting first script...
start wt cmd /k "cd C:\Users\janpr\Desktop\JBG040-Group3 && .venv\Scripts\activate && python -m src.main"

:: Add a 5-second delay before starting the rest
timeout /t 5 /nobreak >nul

:: === LOOP TO RUN REMAINING SCRIPTS ===
set /a count=1
:loop
if !count! lss %runs% (
    echo Starting script !count!...
    start wt cmd /k "cd C:\Users\janpr\Desktop\JBG040-Group3 && .venv\Scripts\activate && python -m src.main"
    set /a count+=1
    goto loop
)

echo All scripts started!
