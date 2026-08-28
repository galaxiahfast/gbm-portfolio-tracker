@echo off
setlocal
cd /d "%~dp0"

py -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
if errorlevel 1 (
    echo Se requiere Python 3.11 o posterior.
    goto :error
)

if not exist ".venv\Scripts\python.exe" (
    echo Preparando el entorno por primera vez...
    py -m venv .venv
    if errorlevel 1 goto :error
)

.venv\Scripts\python.exe -c "import streamlit,pandas,plotly,requests,yfinance,PIL,pytesseract,rapidocr,onnxruntime,reportlab; assert tuple(map(int,streamlit.__version__.split('.')[:2])) >= (1,62)" >nul 2>&1
if errorlevel 1 (
    echo Instalando componentes requeridos...
    .venv\Scripts\python.exe -m pip install --use-feature=truststore --upgrade pip setuptools wheel
    if errorlevel 1 goto :error
    .venv\Scripts\python.exe -m pip install --use-feature=truststore --no-build-isolation -r requirements.txt
    if errorlevel 1 goto :error
)

echo Abriendo Portafolio GBM+...
.venv\Scripts\python.exe -m streamlit run app.py
goto :end

:error
echo.
echo No se pudo preparar la aplicacion. Revisa README.md.
pause

:end
endlocal
