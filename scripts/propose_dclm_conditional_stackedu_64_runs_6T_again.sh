#!/bin/bash


# conditioned on simulated optimal, with 5T web tokens 
: 'for R in 4
do 
    rmc-eval fit -c  src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-dclm-conditional-stackedu-64-6T.yaml \
        -g dfc87a9b \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_collapse_dclm_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none
done 
'

# conditioned on exact optimal, with correct token counts now 
: 'for R in 4
do 
    rmc-eval fit -c  src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-dclm-conditional-stackedu-64-exact-6T.yaml \
        -g e48b584e \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --dashboard mixing-paper \
        --proposer-type exact \
        --kl-reg 0.05 \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_collapse_dclm_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none
done 
'


for R in 4
do
    rmc-eval fit -c  src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-dclm-conditional-stackedu-64-exact-seed-2-6T.yaml \
        -g dab28701 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --dashboard mixing-paper \
        --proposer-type exact \
        --kl-reg 0.05 \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_collapse_dclm_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none

done



