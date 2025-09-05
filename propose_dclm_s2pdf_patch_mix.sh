#!/bin/bash


for R in 3 4 5
do 

    rmc-eval fit -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-s2pdf.yaml \
        -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 83ea88e1 \
        -g 62e7dc06 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 128 \
        --train-split 125
done 
