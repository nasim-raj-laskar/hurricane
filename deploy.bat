@echo off
echo ========================================
echo Hurricane Damage - Astronomer Deployment
echo ========================================
echo.

echo Checking Astro CLI...
astro version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Astro CLI not found!
    echo Install it with: winget install -e --id Astronomer.Astro
    exit /b 1
)

echo.
echo Validating DAGs...
astro dev parse
if errorlevel 1 (
    echo ERROR: DAG validation failed!
    exit /b 1
)

echo.
echo DAGs validated successfully!
echo.
echo Ready to deploy. Run: astro deploy
echo.
echo Make sure you have:
echo   1. Logged in: astro login
echo   2. Updated .astro/deploy.yaml with your workspace ID
echo   3. Set environment variables in Astronomer UI
echo.
pause
