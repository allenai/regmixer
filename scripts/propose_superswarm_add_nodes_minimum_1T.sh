#!/bin/bash

# exact with properly seeded swarms for 1 and 2
for R in 4
do 



    rmc-eval fit -c  src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-add-single-nodes-minimum-exact-1T.yaml \
        -g 02639333 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_add_sources_requested_vs_available_tokens.yaml \
        --requested-tokens 1_000_000_000_000 \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none

    rmc-eval fit -c  src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-add-single-nodes-minimum-exact-seed-1-1T.yaml \
        -g a4c59007 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_add_sources_requested_vs_available_tokens.yaml \
        --requested-tokens 1_000_000_000_000 \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none

    rmc-eval fit -c  src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-add-single-nodes-minimum-exact-seed-2-1T.yaml \
        -g 3f13feb3 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_add_sources_requested_vs_available_tokens.yaml \
        --requested-tokens 1_000_000_000_000 \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none


done 

