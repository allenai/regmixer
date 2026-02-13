#!/bin/bash


# exact, 6T

# sparse
: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-sparse.yaml \
    -g 08590351 \
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
    --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none'

: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-sparse.yaml \
    -g 08590351 \
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
    --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
    --requested-tokens 1_000_000_000_000 \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none'


: 'for SEED in 1 2 
do 
    rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-sparse.yaml \
        -g 08590351 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed $SEED \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
        --repetition-factor 4 \
        --n-test 10 \
        --fit-only
done 
'


: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-sparse.yaml \
    -g 08590351 \
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
    --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none \
    --repetition-factor 4 \
    --fit-only \
    --test-ratios-path cache/c243a53c_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/8dbeb1dc_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/c243a53c_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/8dbeb1dc_pretraining_tasks_for_paper_metrics.pkl
    '


# with recentered mix

# [1T, 
#6T] x [sim, 0.0, 0.01, 0.05]

# 6T x 0.01, 0.05
: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-sparse-fixed.yaml \
    -g da37aa96 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type log_linear \
    --proposer-type exact \
    --kl-reg 0.01 \
    --dashboard mixing-paper \
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none

'

: 'for LAMBDA in 0.01 0.05
do 
    rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-sparse-fixed.yaml \
        -g da37aa96 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg $LAMBDA \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
        --requested-tokens 1_000_000_000_000 \
        --repetition-factor 4 \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none
done '



rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-sparse-fixed.yaml \
    -g da37aa96 \
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
    --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
    --requested-tokens 1_000_000_000_000 \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none \
    --fit-only \
    --test-ratios-path cache/4ea8fa74_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/4a2a6966_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/4ea8fa74_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/4a2a6966_pretraining_tasks_for_paper_metrics.pkl
