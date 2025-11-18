#!/bin/bash

for R in 4
do 
    for SEED in 0 #1 2 
    do 
        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-stackedu-flat.yaml \
            -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -g ee28fc9c \
            -g 2acff647 \
            -g 62e7dc06 \
            -G pretraining_tasks_for_paper \
            -a 1 \
            -S 100_000 \
            -s 1 \
            --opt-avg-metric \
            --seed $SEED \
            --regression-type log_linear \
            --proposer-type exact \
            --kl-reg 0.05 \
            --dashboard regmixer \
            --dashboard mixing-paper \
            --constrain-objective \


            # CAREFUL: THIS IS WITH OLD PES2O COUNT, UED FIXED INSTEAD NEXT TIME

            --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
            --repetition-factor $R \
            --drop-metrics ultrachat_masked_ppl \
            --drop-metrics wildchat_masked_ppl \
            --drop-metrics qasper_yesno:rc::olmes \
            --drop-metrics sciriff_yesno:rc::olmes \
            --drop-metrics lab_bench_dbqa \
            --drop-metrics lab_bench_protocolqa \
            --drop-metrics medqa_en:rc::none \
            --train-split 64 \
            --train-split 64 \
            --train-split 64 \
            --patched
    done
done 