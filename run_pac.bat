
            @echo off
            setlocal enabledelayedexpansion
            
			REM Change the 'username_new' to the username before the git repository (eg. omurphy10)
			set username_new="sungw"
			
            REM Specify the folder path for the log file
            set "LOG_FOLDER=/mnt/c/Users/%username_new%/GitHub/tram_protocol_eeg/src/analysis/pac/logs"
            
            REM Get the current date and time in the desired format
            for /f "tokens=2 delims==" %%I in ('wmic OS Get localdatetime /value') do set "dt=%%I"
            set "YYYY=%dt:~0,4%"
            set "MM=%dt:~4,2%"
            set "DD=%dt:~6,2%"
            set "HH=%dt:~8,2%"
            set "Min=%dt:~10,2%"
            set "SS=%dt:~12,2%"
            set "LOG_FILE=%LOG_FOLDER%/logs.%YYYY%%MM%%DD%-%HH%%Min%%SS%.txt"
            
            set CAFFEINE_PROCESS_NAME=caffeine64.exe
            
            tasklist /fi "imagename eq %CAFFEINE_PROCESS_NAME%" | findstr /i "%CAFFEINE_PROCESS_NAME%" > nul
            
            if %errorlevel% equ 0 (
                    echo Caffeine is already running.
                    ) else (
                    echo Caffeine is not running. Starting it now.
                    start "" "C:\Users\%username_new%\GitHub\tram_protocol_eeg\src\utils\caffeine64.exe"
            )
            
            REM Activate the Anaconda environment and run the Python script
            call C:\Users\%username_new%\Anaconda3\Scripts\activate.bat
            cd C:\Users\%username_new%\GitHub\tram_protocol_eeg
            call conda activate tram_protocol_eeg
            call python -u -m src.analysis.pac.main | wsl tee "%LOG_FILE%"
            
            pause
            