#!/bin/bash


rmc-eval fit -c  src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-dclm-conditional-stackedu-tree-64-exact-seed-2-1T.yaml \
    -g 0980a86c \
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
    --manual-token-constraint-path src/regmixer/eval/superswarm_1T_collapse_dclm_software_development_tree_requested_vs_available_tokens.yaml \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none