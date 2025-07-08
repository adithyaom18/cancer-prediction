@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "C:\Users\K ADITHYA\anaconda3\condabin\conda.bat" activate "c:\Users\K ADITHYA\Documents\streamlit-cancer-predict-main\.conda"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@"c:\Users\K ADITHYA\Documents\streamlit-cancer-predict-main\.conda\python.exe" -Wi -m compileall -q -l -i C:\Users\KADITH~1\AppData\Local\Temp\tmpxl6icp_6 -j 0
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
