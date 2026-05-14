@echo off
cd /d "F:\Argus-Ultimate-main\.claude\worktrees\xenodochial-varahamihira"
C:\Python314\python.exe -m pytest tests/test_system_lifecycle.py -x -q --tb=short > test_results.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> test_results.txt
