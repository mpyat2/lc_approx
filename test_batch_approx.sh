#!/bin/bash

for M in AP WSAP WSL; do
    python3 ila_ap.py \
        --method="$M" \
        --ranges="test_data/lc_split.!" \
        test_data/test_lc.tsv \
        --result "approx_output/${M}-approx.txt" \
        --preview "approx_output/${M}-approx.html"
done
