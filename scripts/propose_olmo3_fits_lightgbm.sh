#!/bin/bash

# hypothesis:  how does fit change with amount of data and dimmension?
: 'for size in 8 16 24 32 40 48
do 
    for seed in 0 1 2 
    do 
        rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense.yaml \
            -g 78dcbdb7 \
            -G pretraining_tasks_for_paper \
            -a 1 \
            -S 100_000 \
            -s 1 \
            --opt-avg-metric \
            --seed $seed \
            --regression-type lightgbm \
            --dashboard mixing-paper \
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
            --repetition-factor 4 \
            --n-test 10 \
            --fit-only \
            --train-split $size
    done 
done
'

: 'for seed in 0 1 2 
do 
    rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense.yaml \
        -g 78dcbdb7 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed $seed \
        --regression-type lightgbm \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
        --repetition-factor 4 \
        --n-test 10 \
        --fit-only
        
done 
'

: 'for seed in 0 1 2 
do 
    rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
        -g 73a6704d \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed $seed \
        --regression-type lightgbm \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
        --repetition-factor 4 \
        --n-test 10 \
        --fit-only
        
done '


for size in 8
do 
    for seed in 1 2 
    do 
        rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
            -g 73a6704d \
            -G pretraining_tasks_for_paper \
            -a 1 \
            -S 100_000 \
            -s 1 \
            --opt-avg-metric \
            --seed $seed \
            --regression-type lightgbm \
            --dashboard mixing-paper \
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
            --repetition-factor 4 \
            --n-test 10 \
            --fit-only \
            --train-split $size
    done 
done

for size in 40 48 #16 24 32 40 48
do 
    for seed in 0 1 2 
    do 
        rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
            -g 73a6704d \
            -G pretraining_tasks_for_paper \
            -a 1 \
            -S 100_000 \
            -s 1 \
            --opt-avg-metric \
            --seed $seed \
            --regression-type lightgbm \
            --dashboard mixing-paper \
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
            --repetition-factor 4 \
            --n-test 10 \
            --fit-only \
            --train-split $size
    done 
done