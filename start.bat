@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Ollama Model Arena: .venv was not found.
  echo Run these commands first:
  echo   py -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install --upgrade pip
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  echo If the "py" launcher is unavailable, use "python" instead.
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" app.py
if errorlevel 1 (
  echo.
  echo Ollama Model Arena exited with an error.
  pause
)
endlocal
