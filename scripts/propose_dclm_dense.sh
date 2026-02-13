#!/bin/bash

: 'rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml \
    -g 0b5a9356 \
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
    --manual-token-constraint-path src/regmixer/eval/cost_ablation_1B_5xC_dclm_requested_vs_available_tokens.yaml \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none
    '


: 'rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml \
    -g 0b5a9356 \
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


: 'rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml \
    -g 0b5a9356 \
    -G pretraining_tasks_for_paper \
    -a 1 \
    -S 100_000 \
    -s 1 \
    --opt-avg-metric \
    --seed 0 \
    --regression-type log_linear \
    --dashboard mixing-paper \
    --constrain-objective \
    --manual-token-constraint-path src/regmixer/eval/cost_ablation_1B_5xC_dclm_requested_vs_available_tokens.yaml \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none '



# assess fit on 50/50 mix of sparse and dense held out mixes.
: 'rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml \
    -g 0b5a9356 \
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
    --test-ratios-path cache/d78491b3_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/bffece5f_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/d78491b3_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/bffece5f_pretraining_tasks_for_paper_metrics.pkl
'


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
    --repetition-factor 4 \
    --fit-only \
    --test-ratios-path cache/d78491b3_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/bffece5f_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/d78491b3_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/bffece5f_pretraining_tasks_for_paper_metrics.pkl
    '


rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml \
    -g 0b5a9356 \
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
    --test-ratios-path cache/fa9e08f5_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/4c3e6013_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/fa9e08f5_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/4c3e6013_pretraining_tasks_for_paper_metrics.pkl


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
    --repetition-factor 4 \
    --fit-only \
    --test-ratios-path cache/fa9e08f5_pretraining_tasks_for_paper_ratios.pkl \
    --test-ratios-path cache/4c3e6013_pretraining_tasks_for_paper_ratios.pkl \
    --test-metrics-path cache/fa9e08f5_pretraining_tasks_for_paper_metrics.pkl \
    --test-metrics-path cache/4c3e6013_pretraining_tasks_for_paper_metrics.pkl
