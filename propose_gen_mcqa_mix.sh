#!/bin/bash


rmc-eval fit -c /Users/rohan/Projects/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mcqa.yaml \
    -g a09b2bf1 \
    -G a09b2bf1_finegrained_evals_v2 \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type linear \
    --workspace ai2-llm/olmo-cookbook \
    --temperature 0.2 \
    --pull-from-dashboard \
    --dashboard olmo3-midtraining-mixing \
    --metric-type primary_score \
    --use-cookbook \
    --custom-name drop_rows_30_47_49 \
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mcqa_requested_vs_available_tokens.yaml \
    --repetition-factor 1 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mcqa-round-5-dummy.yaml \
    --use-reference-model-predicted-scores \



    : '--constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mcqa_requested_vs_available_tokens.yaml \
    --repetition-factor 1 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mcqa-round-5-dummy.yaml \
    --use-reference-model-predicted-scores \
    '