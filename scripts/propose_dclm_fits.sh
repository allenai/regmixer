#!/bin/bash


: 'for SEED in 0 1 2
do
    # different variants of GP 
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 62e7dc06 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed $SEED \
        --regression-type gp \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/cost_ablation_1B_5xC_dclm_requested_vs_available_tokens.yaml \
        --repetition-factor 4 \
        --n-test 10 \
        --fit-only

    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 62e7dc06 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed $SEED \
        --regression-type log_linear \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/cost_ablation_1B_5xC_dclm_requested_vs_available_tokens.yaml \
        --repetition-factor 4 \
        --n-test 10 \
        --fit-only
    
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 62e7dc06 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed $SEED \
        --regression-type lightgbm \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/cost_ablation_1B_5xC_dclm_requested_vs_available_tokens.yaml \
        --repetition-factor 4 \
        --n-test 10 \
        --fit-only \
        --early-stopping 3
done 
'

# do all 128 (127) runs, try gp, log-linear, lightgbm, search
: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type gp \
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
    --drop-metrics medqa_en:rc::none



rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type log_linear \
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
    --drop-metrics medqa_en:rc::none



rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type lightgbm \
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
    --drop-metrics medqa_en:rc::none

'
: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type search \
    --proposer-type search \
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
    --drop-metrics medqa_en:rc::none
'

: 'for SEED in 0 1 2
do
    # test autoscale fit NEED TO REDO
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 62e7dc06 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed $SEED \
        --regression-type autoscale \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/dclm_final_requested_vs_available_tokens.yaml \
        --requested-tokens 6_000_000_000_000 \
        --repetition-factor 4 \
        --n-test 10 \
        --fit-only
done 



for SEED in 0 1 2
do
    # test bimix fit
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 62e7dc06 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed $SEED \
        --regression-type bimix \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/dclm_final_requested_vs_available_tokens.yaml \
        --repetition-factor 4 \
        --n-test 10 \
        --fit-only
done 
'


: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type bimix \
    --proposer-type bimix_exact \
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
    --drop-metrics medqa_en:rc::none
'

: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type autoscale \
    --proposer-type autoscale_exact \
    --kl-reg 0.0 \
    --dashboard mixing-paper \
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/dclm_final_requested_vs_available_tokens.yaml \
    --requested-tokens 6_000_000_000_000 \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none
'


: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type autoscale \
    --dashboard mixing-paper \
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/dclm_final_requested_vs_available_tokens.yaml \
    --requested-tokens 6_000_000_000_000 \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none

rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type bimix \
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
    --drop-metrics medqa_en:rc::none
    '

: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type autoscale \
    --proposer-type autoscale_exact \
    --kl-reg 0.0 \
    --dashboard mixing-paper \
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/dclm_final_requested_vs_available_tokens.yaml \
    --requested-tokens 6_000_000_000_000 \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none
'
: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type autoscale \
    --proposer-type autoscale_exact \
    --kl-reg 0.05 \
    --dashboard mixing-paper \
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/dclm_final_requested_vs_available_tokens.yaml \
    --requested-tokens 6_000_000_000_000 \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none'


rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type bimix \
    --proposer-type bimix_exact \
    --kl-reg 0.0 \
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
    --drop-metrics medqa_en:rc::none