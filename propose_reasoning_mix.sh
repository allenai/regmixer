#!/bin/bash

: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/reasoning-v2.yaml \
    -g e83bd48b \
    -G midtraining_finegrained_evals_v2 \
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
    --temperature 0.2 \
    --fit-only
'

: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/reasoning-v2.yaml \
    -g e83bd48b \
    -G midtraining_code_math_aggregate_evals_v2_for_reasoning_v2 \
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
    --temperature 0.2 \
    --fit-only'




: 'for R in 2 3 4
do 
    rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/reasoning-v2.yaml \
        -g e83bd48b \
        -G midtraining_code_math_aggregate_evals_v2_for_reasoning_v2 \
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
        --temperature 0.2 \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_reasoning_requested_vs_available_tokens_v2_10.yaml \
        --repetition-factor $R \
        --fixed-search-weight '{"gen_mc_code_math": 0}' \

done 
    #--dro-reference-model-id src/regmixer/internal/config/midtraining/reasoning-round-5.yaml \
    #--use-reference-model-predicted-scores
'



: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/reasoning-v2.yaml \
        -g e83bd48b \
        -G midtraining_code_math_aggregate_evals_v2_for_reasoning_v2 \
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
        --temperature 0.2 \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/reasoning-round-5.yaml \
        --use-reference-model-predicted-scores
'


: 'for R in 1 2 3 4
do 
    rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/reasoning-v2.yaml \
        -g e83bd48b \
        -G midtraining_code_math_aggregate_evals_v2_for_reasoning_v2 \
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
        --temperature 0.2 \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_reasoning_requested_vs_available_tokens_v2_10.yaml \
        --repetition-factor $R \
        --fixed-search-weight '{"gen_mc_code_math": 0}' \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/reasoning-round-5.yaml \
        --use-reference-model-predicted-scores
done '


: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/reasoning-v2.yaml \
        -g e83bd48b \
        -G midtraining_code_math_aggregate_evals_v2_for_reasoning_v2 \
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
        --temperature 0.2 \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_reasoning_requested_vs_available_tokens_v2_75.yaml \
        --repetition-factor 1 \
        --fixed-search-weight '{"gen_mc_code_math": 0}' \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/reasoning-round-5.yaml \
        --use-reference-model-predicted-scores \
        --use-reference-model-as-search-prior'




rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/reasoning-v2.yaml \
        -g e83bd48b \
        -G midtraining_code_math_aggregate_evals_v2_for_reasoning_v2 \
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
        --temperature 0.2 \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_reasoning_requested_vs_available_tokens_v2_15.yaml \
        --repetition-factor 1 \
        --fixed-search-weight '{"gen_mc_code_math": 0}' \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/reasoning-round-5.yaml \
        --use-reference-model-predicted-scores \
        --use-reference-model-as-search-prior
