#!/bin/bash

# NB: need to put the config/exp group that has the superset of domains first! 



# repetition only / unconstrained
: 'for R in 2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G a3e06472_515eaf2d_finegrained_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R
done'


: 'rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G a3e06472_515eaf2d_finegrained_evals_v2 \
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
        --custom-name drop_outliers
'


# pareto w.r.t round 5
: 'for R in 1 2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G a3e06472_515eaf2d_finegrained_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-natural.yaml \
        --use-reference-model-predicted-scores
done


rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G a3e06472_515eaf2d_finegrained_evals_v2 \
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
        --custom-name drop_outliers \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-natural.yaml \
        --use-reference-model-predicted-scores \
'

: 'rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --fit-only
'




# code and math only 
: 'for R in 1 2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G a3e06472_515eaf2d_code_math_finegrained_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R
done


rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G a3e06472_515eaf2d_code_math_finegrained_evals_v2 \
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
        --custom-name drop_outliers
'

: 'for R in 1 2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G a3e06472_515eaf2d_code_math_finegrained_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-natural.yaml \
        --use-reference-model-predicted-scores
done


rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G a3e06472_515eaf2d_code_math_finegrained_evals_v2 \
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
        --custom-name drop_outliers \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-natural.yaml \
        --use-reference-model-predicted-scores \
'

# coarse 
: 'for R in 1 2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R
done


rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_aggregate_evals_v2 \
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
        --custom-name drop_outliers
'


: 'for R in 1 2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-natural.yaml \
        --use-reference-model-predicted-scores
done


rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-natural.yaml \
        --use-reference-model-predicted-scores \
'

# play with tolerance 
: 'for R in 1 #2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G a3e06472_515eaf2d_code_math_finegrained_evals_v2 \
        -a 1 \
        -S 1_000_000 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-round-5.yaml \
        --use-reference-model-predicted-scores \
        --tol 0.3
done'







: 'for R in 1 2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_code_math_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R
done


rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_code_math_aggregate_evals_v2 \
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
        --custom-name drop_outliers


for R in 1 2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_code_math_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-round-5.yaml \
        --use-reference-model-predicted-scores
done


rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_code_math_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-round-5.yaml \
        --use-reference-model-predicted-scores
'


: 'for R in 1 2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --obj-weights midtraining_aggregate_evals_v2_weights_7_3
done


rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --obj-weights midtraining_aggregate_evals_v2_weights_7_3


for R in 1 2 3 4 
do 
    rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens.yaml \
        --repetition-factor $R \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-round-5.yaml \
        --use-reference-model-predicted-scores \
        --obj-weights midtraining_aggregate_evals_v2_weights_7_3
done


rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G midtraining_aggregate_evals_v2 \
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
        --custom-name drop_outliers \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-round-5.yaml \
        --use-reference-model-predicted-scores \
        --obj-weights midtraining_aggregate_evals_v2_weights_7_3
'



: 'for  T in 30 35 65 #40 45 50 55 60 65 70
do 
     rmc-eval fit -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code-and-math.yaml \
        -c /weka/oe-data-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/code.yaml \
        -g a3e06472 \
        -g 515eaf2d \
        -G a3e06472_515eaf2d_code_math_finegrained_evals_v2 \
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
        --custom-name drop_outliers \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/midtraining_code_math_requested_vs_available_tokens_$T.yaml \
        --repetition-factor 1 \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/code-math-round-5.yaml \
        --use-reference-model-predicted-scores \
        --fixed-search-weight '{"gen-mcqa": 0}' \
        --temperature 0.2
        #--use-reference-model-as-search-prior \
        #--temperature 0.2
done '