#!/bin/bash

# the right one 
for R in 4
do
    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest_FIXED_PES2O.yaml \
        --repetition-factor $R \
        --neighborhood 5xC-30m-superswarm-ee28fc9c-0485 \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 256
    '

    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --dashboard regmixer \
        --proposer-type exact \
        --kl-reg 0.05 \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest_FIXED_PES2O.yaml \
        --repetition-factor $R \
        --neighborhood 5xC-30m-superswarm-ee28fc9c-0485 \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 256



done





