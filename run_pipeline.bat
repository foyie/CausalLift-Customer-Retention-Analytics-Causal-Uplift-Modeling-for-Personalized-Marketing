@echo off
REM Customer Retention Analytics - Automated Pipeline for Windows
REM Run the entire analysis in sequence

setlocal enabledelayedexpansion

echo.
echo ========================================================
echo      Customer Retention Analytics - Pipeline
echo           Causal Inference for Marketing
echo ========================================================
echo.

REM Configuration
set "DATA_DIR=data"
set "RAW_DATA=%DATA_DIR%\raw\synthetic_retail.csv"
set "PROCESSED_DATA=%DATA_DIR%\processed\features_engineered.csv"
set "MODELS_DIR=models"
set "RESULTS_DIR=results"
set "N_SAMPLES=100000"
set "RANDOM_SEED=42"

REM ============================================================================
REM PHASE 1: SETUP
REM ============================================================================
echo.
echo [1/5] PHASE 1: Setup and Validation
echo.

echo Creating directory structure...
if not exist "%DATA_DIR%\raw" mkdir "%DATA_DIR%\raw"
if not exist "%DATA_DIR%\processed" mkdir "%DATA_DIR%\processed"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "logs" mkdir "logs"
if not exist "app" mkdir "app"
echo [OK] Directories created
echo.

echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ and add to PATH
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
echo [OK] Python %PYTHON_VERSION%
echo.

echo Checking required packages...
python -c "import econml, xgboost, pandas, numpy, streamlit" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some packages missing. Run: pip install -r requirements.txt
    echo Attempting to install...
    pip install -r requirements.txt
)
echo [OK] Packages verified
echo.

REM ============================================================================
REM PHASE 2: DATA GENERATION
REM ============================================================================
echo [2/5] PHASE 2: Data Generation
echo.
echo Generating %N_SAMPLES% customer records...
echo.

python src/data_generation.py ^
    --output "%RAW_DATA%" ^
    --n_samples %N_SAMPLES% ^
    --seed %RANDOM_SEED%

if not exist "%RAW_DATA%" (
    echo [ERROR] Data generation failed
    pause
    exit /b 1
)

for %%A in ("%RAW_DATA%") do set "FILE_SIZE=%%~zA"
echo [OK] Data generation completed (%FILE_SIZE% bytes)
echo.

REM ============================================================================
REM PHASE 3: FEATURE ENGINEERING
REM ============================================================================
echo [3/5] PHASE 3: Feature Engineering
echo.
echo Creating 80+ features from raw data...
echo.

python src/feature_engineering.py ^
    --input "%RAW_DATA%" ^
    --output "%PROCESSED_DATA%" ^
    --preprocessor "%MODELS_DIR%\preprocessor.pkl"

if not exist "%PROCESSED_DATA%" (
    echo [ERROR] Feature engineering failed
    pause
    exit /b 1
)

for %%A in ("%PROCESSED_DATA%") do set "FILE_SIZE=%%~zA"
echo [OK] Feature engineering completed (%FILE_SIZE% bytes)
echo.

REM ============================================================================
REM PHASE 4: CAUSAL ESTIMATION
REM ============================================================================
echo [4/5] PHASE 4: Causal Model Training
echo.
echo Training T-Learner, X-Learner, Causal Forest, and Ensemble...
echo.

python src/causal_estimation.py ^
    --data "%PROCESSED_DATA%" ^
    --output "%MODELS_DIR%"

if not exist "%MODELS_DIR%\causal_summary.csv" (
    echo [ERROR] Causal estimation failed
    pause
    exit /b 1
)

echo [OK] Causal estimation completed
echo.
echo Causal Model Summary:
type "%MODELS_DIR%\causal_summary.csv"
echo.

REM ============================================================================
REM PHASE 5: VALIDATION
REM ============================================================================
echo [5/5] PHASE 5: CATE Validation
echo.
echo Running Qini curves, AUUC, and placebo tests...
echo.

python src/validation.py ^
    --data "%PROCESSED_DATA%" ^
    --output "%RESULTS_DIR%"

if not exist "%RESULTS_DIR%\validation_results.json" (
    echo [ERROR] Validation failed
    pause
    exit /b 1
)

echo [OK] Validation completed
echo.

REM ============================================================================
REM PHASE 6: SUMMARY
REM ============================================================================
echo.
echo ========================================================
echo              ✓ PIPELINE COMPLETE
echo ========================================================
echo.

echo Generated Artifacts:
echo   [*] Raw Data:       %RAW_DATA%
echo   [*] Processed Data: %PROCESSED_DATA%
echo   [*] Models:         %MODELS_DIR%\
echo   [*] Results:        %RESULTS_DIR%\
echo.

echo Next Steps:
echo.
echo 1. LAUNCH DASHBOARD:
echo    ^> streamlit run app/dashboard.py
echo.
echo 2. NAVIGATE TO:
echo    ^> http://localhost:8501
echo.
echo 3. EXPLORE PAGES:
echo    - Overview: KPIs and architecture
echo    - CATE Analysis: Distribution and segments
echo    - Model Comparison: Performance metrics
echo    - Policy Simulator: Interactive targeting
echo    - Validation: Statistical tests
echo.

echo Useful Commands:
echo.
echo   Check model summary:
echo   ^> type "%MODELS_DIR%\causal_summary.csv"
echo.
echo   View high responders:
echo   ^> type "%RESULTS_DIR%\high_responders.csv"
echo.

echo 🚀 Ready to explore your Customer Retention Analytics!
echo.
echo For detailed information, see: SETUP_GUIDE.md
echo.

pause
