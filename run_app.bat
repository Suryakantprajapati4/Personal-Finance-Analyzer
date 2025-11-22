@echo off
REM ==== Activate the venv in C:\Users\surya\.venv ====
call "C:\Users\surya\.venv\Scripts\activate.bat"

REM ==== Run the Streamlit app ====
python -m streamlit run "C:\Project\python\app.py"

echo.
echo -------------------------------
echo If there was an error above, press any key to close...
pause >nul
