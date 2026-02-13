#!/bin/bash

: 'rmc-eval fit -c  src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-s2pdfv1-partitioned-partial-exact-seed-0-1T.yaml \
    -g 3b51e0ee \
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
    --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_partition_s2pdf_requested_vs_available_tokens_partial.yaml \
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

: 'rmc-eval fit -c  src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-s2pdfv1-partitioned-partial-exact-seed-1-1T.yaml \
    -g a6fa7759 \
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
    --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_partition_s2pdf_requested_vs_available_tokens_partial.yaml \
    --requested-tokens 1_000_000_000_000 \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none'

rmc-eval fit -c  src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-s2pdfv1-partitioned-partial-exact-seed-2-1T.yaml \
    -g 3f57a731 \
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
    --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_partition_s2pdf_requested_vs_available_tokens_partial.yaml \
    --requested-tokens 1_000_000_000_000 \
    --repetition-factor 4 \
    --drop-metrics ultrachat_masked_ppl \
    --drop-metrics wildchat_masked_ppl \
    --drop-metrics qasper_yesno:rc::olmes \
    --drop-metrics sciriff_yesno:rc::olmes \
    --drop-metrics lab_bench_dbqa \
    --drop-metrics lab_bench_protocolqa \
    --drop-metrics medqa_en:rc::none


