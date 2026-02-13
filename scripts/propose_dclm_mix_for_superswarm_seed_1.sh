#!/bin/bash


: 'for R in 4
do 
    for size in 25
    do 
        for seed in 1
        do
            rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
                -g 62e7dc06 \
                -G pretraining_tasks_for_paper \
                -a 1 \
                -S 100_000 \
                -s 1 \
                --opt-avg-metric \
                --seed $seed \
                --regression-type log_linear \
                --proposer-type exact \
                --kl-reg 0.05 \
                --dashboard mixing-paper \
                --constrain-objective \
                --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_dclm_only_requested_vs_available_tokens.yaml \
                --repetition-factor $R \
                --train-split $size \
                --drop-metrics ultrachat_masked_ppl \
                --drop-metrics wildchat_masked_ppl \
                --drop-metrics qasper_yesno:rc::olmes \
                --drop-metrics sciriff_yesno:rc::olmes \
                --drop-metrics lab_bench_dbqa \
                --drop-metrics lab_bench_protocolqa \
                --drop-metrics medqa_en:rc::none
        done 
    done
done
'

for R in 4
do 
    for size in 64 32 25
    do 
        for seed in 1
        do
            rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
                -g 62e7dc06 \
                -G pretraining_tasks_for_paper \
                -a 1 \
                -S 100_000 \
                -s 1 \
                --opt-avg-metric \
                --seed $seed \
                --regression-type log_linear \
                --proposer-type exact \
                --kl-reg 0.05 \
                --dashboard mixing-paper \
                --constrain-objective \
                --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_dclm_only_requested_vs_available_tokens.yaml \
                --requested-tokens 1_000_000_000_000 \
                --repetition-factor $R \
                --train-split $size \
                --drop-metrics ultrachat_masked_ppl \
                --drop-metrics wildchat_masked_ppl \
                --drop-metrics qasper_yesno:rc::olmes \
                --drop-metrics sciriff_yesno:rc::olmes \
                --drop-metrics lab_bench_dbqa \
                --drop-metrics lab_bench_protocolqa \
                --drop-metrics medqa_en:rc::none
        done 
    done
done