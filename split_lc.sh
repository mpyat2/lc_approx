#!/bin/bash

_PERIOD=0.40903915
_EPOCH=2460030.023
_START_PHASE=-0.2
_STOP_PHASE=0.2

python3 split_lc.py test_data/test_lc.tsv --output test_data/lc_split.! --preview test_data/lc_split.html --epoch $_EPOCH --period $_PERIOD --start-phase $_START_PHASE --stop-phase $_STOP_PHASE
