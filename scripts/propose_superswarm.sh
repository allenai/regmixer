#!/bin/bash

# the right one 
for R in 4
do
    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G qa_tasks \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R 

    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G math_tasks \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R 
    '

    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R '


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
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_1T_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R 


done
