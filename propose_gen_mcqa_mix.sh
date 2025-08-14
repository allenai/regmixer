#!/bin/bash



: 'rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mcqa.yaml \
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
    --fit-only'

: 'rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mcqa.yaml \
    -g a09b2bf1 \
    -G a09b2bf1_non_code_math_finegrained_evals_v2 \
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
    --repetition-factor 4 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mcqa-round-5-dummy.yaml \
    --use-reference-model-predicted-scores

rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mcqa.yaml \
    -g a09b2bf1 \
    -G a09b2bf1_non_code_math_finegrained_evals_v2 \
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
    --use-reference-model-predicted-scores
'
# THE ONE WE USED
: 'rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mcqa.yaml \
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
    --custom-name drop_rows_30_47_49_fixed_repetition_constraint_bug \
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mcqa_requested_vs_available_tokens_granular.yaml \
    --repetition-factor 1 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mcqa-round-5-dummy.yaml \
    --use-reference-model-predicted-scores \
'
###########


: 'rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mcqa.yaml \
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
    --repetition-factor 4 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mcqa-round-5-dummy.yaml \
    --use-reference-model-predicted-scores \'



for  R in 1 #2 3 4
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mcqa.yaml \
        -g a09b2bf1 \
        -G a09b2bf1_finegrained_evals_v2 \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type linear \
        --workspace ai2-llm/olmo-cookbook \
        --pull-from-dashboard \
        --dashboard olmo3-midtraining-mixing \
        --metric-type primary_score \
        --use-cookbook \
        --custom-name new_coarse_reproduce_${R} \
        --temperature 0.2 \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mcqa_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mcqa-round-5-dummy.yaml \
        --use-reference-model-predicted-scores
done 


#         
