#!/bin/bash


# sim, 6T
: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense.yaml \
    -g 78dcbdb7 \
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


# sim 1T    
rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense.yaml \
    -g 78dcbdb7 \
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



# exact, 6T 0.01
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


# exact 1T 0.01
rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense.yaml \
    -g 78dcbdb7 \
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
    --kl-reg 0.03 \
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

rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-dense.yaml \
    -g 78dcbdb7 \
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



# try with sparse swarm
for LAMBDA in 0.0 0.01
do 
    rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-sparse.yaml \
        -g 08590351 \
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
        --repetition-factor 4 \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none


    rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/olmo3-5xC-30m-sparse.yaml \
        -g 08590351 \
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
done 
