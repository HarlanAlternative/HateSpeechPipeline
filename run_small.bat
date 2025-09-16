@echo off
setlocal
set PY=python

echo Starting Reddit Data Analysis Pipeline (SMALL MODE)
echo ==================================================

%PY% scripts/00_env_check.py --config configs/exp_small.yaml || goto :eof
echo.
echo Environment check completed successfully!
echo.

%PY% scripts/01_slice_dataset.py --config configs/exp_small.yaml || goto :eof
echo.
echo Dataset slicing completed successfully!
echo.

%PY% scripts/02_verify_parquet.py --config configs/exp_small.yaml || goto :eof
echo.
echo Parquet verification completed successfully!
echo.

%PY% scripts/03_prepare_corpus.py --config configs/exp_small.yaml || goto :eof
echo.
echo Corpus preparation completed successfully!
echo.

%PY% scripts/04_baseline_tfidf_lr.py --config configs/exp_small.yaml || goto :eof
echo.
echo Baseline model training completed successfully!
echo.

%PY% scripts/05_build_graph.py --config configs/exp_small.yaml || goto :eof
echo.
echo Graph construction completed successfully!
echo.

%PY% scripts/06_visualize_diffusion.py --config configs/exp_small.yaml || goto :eof
echo.
echo Diffusion visualization completed successfully!
echo.

%PY% scripts/07_eval_report.py --config configs/exp_small.yaml || goto :eof
echo.
echo Evaluation report completed successfully!
echo.

echo ==================================================
echo DONE (SMALL MODE) - All analysis completed!
echo Check artifacts/ and figures/ directories for results.
echo ==================================================
endlocal
