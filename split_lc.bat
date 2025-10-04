@echo off

SET PERIOD=0.40903915
SET EPOCH=2460030.023
SET START_PHASE=-0.2
SET STOP_PHASE=0.2

split_lc.py test_data\test_lc.tsv --output test_data\lc_split.! --preview test_data\lc_split.html --epoch %EPOCH% --period %PERIOD% --start-phase %START_PHASE% --stop-phase %STOP_PHASE%
pause
