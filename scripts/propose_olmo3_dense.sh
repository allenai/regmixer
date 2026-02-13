#!/bin/bash


# exact, 6T

# dense
: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense.yaml \
    -g 78dcbdb7 \
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


    
: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense.yaml \
    -g 78dcbdb7 \
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
    --drop-metrics medqa_en:rc::none
'

    
: 'for SEED in 0 1 2 
do

    rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense.yaml \
        -g 78dcbdb7 \
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


: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense.yaml \
    -g 78dcbdb7 \
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
    --manual-token-constraint-path src/regmixer/eval/dclm_final_requested_vs_available_tokens.yaml \
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
    --test-metrics-path cache/8dbeb1dc_pretraining_tasks_for_paper_metrics.pkl'




# fixed swawrm 


# 0.05 6T
: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
    -g 73a6704d \
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
    --drop-metrics medqa_en:rc::none

# 0.01 6T
rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
    -g 73a6704d \
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


# 0.05 1T
: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
    -g 73a6704d \
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
    --drop-metrics medqa_en:rc::none

# 0.01 1T
rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
    -g 73a6704d \
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
    --requested-tokens 1_000_000_000_000 \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none
'

# Simulated 6T
: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
    -g 73a6704d \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type log_linear \
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

rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
    -g 73a6704d \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type log_linear \
    --proposer-type exact \
    --kl-reg 0.0 \
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

# regression fit 
: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
    -g 73a6704d \
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
    --fit-only \
    --test-ratios-path cache/4ea8fa74_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/4a2a6966_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/4ea8fa74_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/4a2a6966_pretraining_tasks_for_paper_metrics.pkl
'


rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
    -g 73a6704d \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type log_linear \
    --dashboard mixing-paper \
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
    --repetition-factor 4 \
    --requested-tokens 1_000_000_000_000 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none

rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense-fixed.yaml \
    -g 73a6704d \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type log_linear \
    --proposer-type exact \
    --kl-reg 0.0 \
    --dashboard mixing-paper \
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/olmo3_sources_requested_vs_available_tokens.yaml \
    --repetition-factor 4 \
    --requested-tokens 1_000_000_000_000 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none
