#!/bin/bash



# the right one 
for R in 4
do

    for SEED in 1 
    do 
        : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -g ee28fc9c \
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
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
            --requested-tokens 1_000_000_000_000 \
            --repetition-factor $R \
            --drop-metrics ultrachat_masked_ppl \
            --drop-metrics wildchat_masked_ppl \
            --drop-metrics qasper_yesno:rc::olmes \
            --drop-metrics sciriff_yesno:rc::olmes \
            --drop-metrics lab_bench_dbqa \
            --drop-metrics lab_bench_protocolqa \
            --drop-metrics medqa_en:rc::none \
            --train-split 128
'

        : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -g ee28fc9c \
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
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
            --repetition-factor $R \
            --drop-metrics ultrachat_masked_ppl \
            --drop-metrics wildchat_masked_ppl \
            --drop-metrics qasper_yesno:rc::olmes \
            --drop-metrics sciriff_yesno:rc::olmes \
            --drop-metrics lab_bench_dbqa \
            --drop-metrics lab_bench_protocolqa \
            --drop-metrics medqa_en:rc::none \
            --train-split 128
        '

        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -g ee28fc9c \
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
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
            --repetition-factor $R \
            --drop-metrics ultrachat_masked_ppl \
            --drop-metrics wildchat_masked_ppl \
            --drop-metrics qasper_yesno:rc::olmes \
            --drop-metrics sciriff_yesno:rc::olmes \
            --drop-metrics lab_bench_dbqa \
            --drop-metrics lab_bench_protocolqa \
            --drop-metrics medqa_en:rc::none \
            --train-split 65

    done


done 


