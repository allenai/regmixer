#!/bin/bash



: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/30m-5xC-dclm-centered-0.05-exact-6T.yaml \
    -g 1f734674 \
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
    --drop-metrics medqa_en:rc::none'


: 'for SEED in 0 1 2
do
    rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/30m-5xC-dclm-centered-0.05-exact-6T.yaml \
        -g 1f734674 \
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
        --n-test 10 \
        --fit-only
done 
'

: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/30m-5xC-dclm-centered-0.05-exact-6T.yaml \
    -g 1f734674 \
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
    --natural-kl
'

# centered on strong prior
: 'rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/30m-5xC-dclm-centered-0.05-exact-6T.yaml \
    -g 1f734674 \
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
    --test-ratios-path cache/bffece5f_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/18e9da22_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/d8eb4902_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/bffece5f_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/18e9da22_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/d8eb4902_pretraining_tasks_for_paper_metrics.pkl


# centered on natural
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
    --test-ratios-path cache/bffece5f_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/18e9da22_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/d8eb4902_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/bffece5f_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/18e9da22_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/d8eb4902_pretraining_tasks_for_paper_metrics.pkl
'


rmc-eval fit -c src/regmixer/config/for_paper/contribution_1/30m-5xC-dclm-centered-0.05-exact-6T.yaml \
    -g 1f734674 \
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
    --test-ratios-path cache/bffece5f_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/1184a19d_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/04d827ea_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/bffece5f_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/1184a19d_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/04d827ea_pretraining_tasks_for_paper_metrics.pkl \



# centered on natural
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
    --test-ratios-path cache/bffece5f_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/1184a19d_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/04d827ea_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/bffece5f_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/1184a19d_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/04d827ea_pretraining_tasks_for_paper_metrics.pkl \
