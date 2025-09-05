#!/bin/bash

: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -g a3d4f82c \
    -G midtraining_aggregate_evals_v2 \
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
    #--fit-only
'


: 'for R in 1 2 3 4
do 
    rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
        -g a3d4f82c \
        -G midtraining_aggregate_evals_v2 \
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
        --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_100.yaml \
        --repetition-factor $R
done 
'

: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -g a3d4f82c \
    -G midtraining_aggregate_evals_v2 \
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
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mc-code-math-round-5.yaml \
    --use-reference-model-predicted-scores \


for R in 1 2 3 4
do 
    rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
        -g a3d4f82c \
        -G midtraining_aggregate_evals_v2 \
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
        --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_100.yaml \
        --repetition-factor $R \
        --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mc-code-math-round-5.yaml \
        --use-reference-model-predicted-scores \

done 
'

: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -g a3d4f82c \
    -G midtraining_aggregate_evals_v2 \
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
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_100.yaml \
    --repetition-factor 1 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mc-code-math-round-5.yaml \
    --use-reference-model-predicted-scores \
    --tol 0.75 \
    --use-reference-model-as-search-prior \
    #--temperature 0.2 \
#    --use-reference-model-as-search-prior \
'


: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -g a3d4f82c \
    -G midtraining_aggregate_evals_v2 \
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
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_525_40.yaml \
    --repetition-factor 1 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mc-code-math-round-5.yaml \
    --use-reference-model-predicted-scores \
    --tol 0.25 \
    --use-reference-model-as-search-prior \
    #--temperature 0.2 \
'



# Only optimize w.r.t. code/math tasks, in preparation for code/math/reasoning only macroanneal
: 'for R in 1 2 3 4
do 
rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -g a3d4f82c \
    -G a3d4f82c_code_math_aggregate_evals_v2 \
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
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_85_with_updated_swallowcode_code_fim.yaml \
    --repetition-factor $R \
    --use-reference-model-predicted-scores \
    --temperature 0.2 \
    --fixed-search-weight '{"hqweb-pdf": 0.2, "flan": 0, "instruction-new-format": 0, "nemotron-synth-qa": 0, "rcqa": 0, "reddit-high": 0, "sponge": 0}' \

done '

: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -g a3d4f82c \
    -G a3d4f82c_code_math_aggregate_evals_v2 \
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
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_85_with_updated_swallowcode_code_fim.yaml \
    --repetition-factor 1 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mc-code-math-round-5.yaml \
    --use-reference-model-predicted-scores \
    --use-reference-model-as-search-prior \
    --fixed-search-weight '{"hqweb-pdf": 0.23529411764705882, "flan": 0, "instruction-new-format": 0, "nemotron-synth-qa": 0, "rcqa": 0, "reddit-high": 0, "sponge": 0}'
'




: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -g a3d4f82c \
    -G a3d4f82c_code_math_aggregate_evals_v2 \
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
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_85_with_updated_swallowcode_code_fim_swallowmath.yaml \
    --repetition-factor 1 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mc-code-math-round-5.yaml \
    --use-reference-model-predicted-scores \
    --fixed-search-weight '{"hqweb-pdf": 0.23529411764705882, "flan": 0, "instruction-new-format": 0, "nemotron-synth-qa": 0, "rcqa": 0, "reddit-high": 0, "sponge": 0}' \
    --temperature 0.2 \
    --use-reference-model-as-search-prior \
    --tol -0.75'


: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -g a3d4f82c \
    -G a3d4f82c_code_math_aggregate_evals_v2 \
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
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_85_with_updated_swallowcode_code_fim_swallowmath.yaml \
    --repetition-factor 1 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mc-code-math-round-5.yaml \
    --use-reference-model-predicted-scores \
    --use-reference-model-as-search-prior \
    --fixed-search-weight '{"hqweb-pdf": 0.23529411764705882, "flan": 0, "instruction-new-format": 0, "nemotron-synth-qa": 0, "rcqa": 0, "reddit-high": 0, "sponge": 0}' \
'



: 'rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm-part-2.yaml \
    -g a3d4f82c \
    -g 880f1b76 \
    -G midtraining_aggregate_evals_v2 \
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
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_100.yaml \
    --repetition-factor 1


rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm-part-2.yaml \
    -g a3d4f82c \
    -g 880f1b76 \
    -G midtraining_aggregate_evals_v2 \
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
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_100.yaml \
    --repetition-factor 1 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mc-code-math-round-5.yaml \
    --use-reference-model-predicted-scores \
    --use-reference-model-as-search-prior \
    --tol 0.75
'


rmc-eval fit -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm.yaml \
    -c /weka/oe-training-default/mayeec/olmo-cookbook/src/cookbook/recipes/olmo3-midtraining/mayeec_swarms/round_2/gen-mc-code-math-superswarm-part-2.yaml \
    -g a3d4f82c \
    -g 880f1b76 \
    -G midtraining_aggregate_evals_v2 \
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
    --manual-token-constraint-path src/regmixer/eval/midtraining_gen_mc_code_math_requested_vs_available_tokens_100.yaml \
    --repetition-factor 1 \
    --dro-reference-model-id src/regmixer/internal/config/midtraining/gen-mc-code-math-natural.yaml \
    --use-reference-model-predicted-scores \
    --use-reference-model-as-search-prior \
    --tol -0.25
