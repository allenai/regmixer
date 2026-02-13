#!/bin/bash


: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -A pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
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
    --drop-metrics medqa_en:rc::none 
'

: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -A pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
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

rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -A pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
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

# test fit of aggregation models
: 'for SEED in 0 1 2 
do 
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 62e7dc06 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --seed $SEED \
        --regression-type log_linear \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/dclm_final_requested_vs_available_tokens.yaml \
        --repetition-factor 4 \
        --n-test 10 \
        --fit-only 

    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 62e7dc06 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --seed $SEED \
        --regression-type lightgbm \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/dclm_final_requested_vs_available_tokens.yaml \
        --repetition-factor 4 \
        --n-test 10 \
        --fit-only 

    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 62e7dc06 \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
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

rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
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
    --aggregate-task-families \
    --obj-weights task_families


: 'for SEED in 0 1 2 
do
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 62e7dc06 \
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
        --manual-token-constraint-path src/regmixer/eval/dclm_final_requested_vs_available_tokens.yaml \
        --repetition-factor 4 \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --aggregate-task-families \
        --obj-weights task_families \
        --n-test 10 \
        --fit-only

    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g 62e7dc06 \
        -A pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --seed $SEED \
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
        --n-test 10 \
        --fit-only \

done'




: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
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
    --aggregate-task-families \
    --obj-weights task_families \
    --fit-only \
    --test-ratios-path cache/4c3e6013_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/4c3e6013_pretraining_tasks_for_paper_metrics.pkl \

rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
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
    --fit-only \
    --test-ratios-path cache/4c3e6013_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/4c3e6013_pretraining_tasks_for_paper_metrics.pkl \
'

: 'rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
    -A pretraining_tasks_for_paper \
    --opt-avg-metric \
    -a 1 \
    -S 100_000 \
    -s 1 \
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
    --fit-only \
    --test-ratios-path cache/4c3e6013_avg_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/0c2fa116_avg_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/f2c11abd_avg_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/4c3e6013_avg_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/0c2fa116_avg_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/f2c11abd_avg_pretraining_tasks_for_paper_metrics.pkl \




rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
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
    --fit-only \
    --test-ratios-path cache/4c3e6013_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/0c2fa116_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/f2c11abd_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/4c3e6013_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/0c2fa116_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/f2c11abd_pretraining_tasks_for_paper_metrics.pkl \


rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
    -g 62e7dc06 \
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
    --aggregate-task-families \
    --obj-weights task_families \
    --fit-only \
    --test-ratios-path cache/4c3e6013_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/0c2fa116_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/f2c11abd_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/4c3e6013_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/0c2fa116_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/f2c11abd_pretraining_tasks_for_paper_metrics.pkl \

'