#!/bin/bash

: 'for R in 4
do 
    for SEED in 0 1 2
    do 

        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -c src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-remove-algebraicstack-128-exact-opt-prior-6T.yaml \
            -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-stackedu-flat.yaml \
            -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -g ee28fc9c \
            -g 914e1003 \
            -g 2acff647 \
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
            --dashboard regmixer \
            --dashboard mixing-paper \
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
            --repetition-factor $R \
            --drop-metrics ultrachat_masked_ppl \
            --drop-metrics wildchat_masked_ppl \
            --drop-metrics qasper_yesno:rc::olmes \
            --drop-metrics sciriff_yesno:rc::olmes \
            --drop-metrics lab_bench_dbqa \
            --drop-metrics lab_bench_protocolqa \
            --drop-metrics medqa_en:rc::none \
            --train-split 123 \
            --train-split 5 \
            --train-split 64 \
            --train-split 64 \
            --patched

        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -c src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-remove-algebraicstack-128-exact-opt-prior-6T.yaml \
            -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-stackedu-flat.yaml \
            -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -g ee28fc9c \
            -g 914e1003 \
            -g 2acff647 \
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
            --dashboard regmixer \
            --dashboard mixing-paper \
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
            --requested-tokens 1_000_000_000_000 \
            --repetition-factor $R \
            --drop-metrics ultrachat_masked_ppl \
            --drop-metrics wildchat_masked_ppl \
            --drop-metrics qasper_yesno:rc::olmes \
            --drop-metrics sciriff_yesno:rc::olmes \
            --drop-metrics lab_bench_dbqa \
            --drop-metrics lab_bench_protocolqa \
            --drop-metrics medqa_en:rc::none \
            --train-split 123 \
            --train-split 5 \
            --train-split 64 \
            --train-split 64 \
            --patched

        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -c src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-remove-algebraicstack-128-exact-opt-prior-6T.yaml \
            -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-stackedu-flat.yaml \
            -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -g ee28fc9c \
            -g 914e1003 \
            -g 2acff647 \
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
            --dashboard regmixer \
            --dashboard mixing-paper \
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
            --repetition-factor $R \
            --drop-metrics ultrachat_masked_ppl \
            --drop-metrics wildchat_masked_ppl \
            --drop-metrics qasper_yesno:rc::olmes \
            --drop-metrics sciriff_yesno:rc::olmes \
            --drop-metrics lab_bench_dbqa \
            --drop-metrics lab_bench_protocolqa \
            --drop-metrics medqa_en:rc::none \
            --train-split 59 \
            --train-split 5 \
            --train-split 32 \
            --train-split 32 \
            --patched

        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -c src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-remove-algebraicstack-128-exact-opt-prior-6T.yaml \
            -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-stackedu-flat.yaml \
            -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -g ee28fc9c \
            -g 914e1003 \
            -g 2acff647 \
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
            --dashboard regmixer \
            --dashboard mixing-paper \
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
            --requested-tokens 1_000_000_000_000 \
            --repetition-factor $R \
            --drop-metrics ultrachat_masked_ppl \
            --drop-metrics wildchat_masked_ppl \
            --drop-metrics qasper_yesno:rc::olmes \
            --drop-metrics sciriff_yesno:rc::olmes \
            --drop-metrics lab_bench_dbqa \
            --drop-metrics lab_bench_protocolqa \
            --drop-metrics medqa_en:rc::none \
            --train-split 59 \
            --train-split 5 \
            --train-split 32 \
            --train-split 32 \
            --patched

        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -c src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-remove-algebraicstack-128-exact-opt-prior-6T.yaml \
            -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-stackedu-flat.yaml \
            -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -g ee28fc9c \
            -g 914e1003 \
            -g 2acff647 \
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
            --dashboard regmixer \
            --dashboard mixing-paper \
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
            --repetition-factor $R \
            --drop-metrics ultrachat_masked_ppl \
            --drop-metrics wildchat_masked_ppl \
            --drop-metrics qasper_yesno:rc::olmes \
            --drop-metrics sciriff_yesno:rc::olmes \
            --drop-metrics lab_bench_dbqa \
            --drop-metrics lab_bench_protocolqa \
            --drop-metrics medqa_en:rc::none \
            --train-split 20 \
            --train-split 5 \
            --train-split 15 \
            --train-split 25 \
            --patched

        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -c src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-remove-algebraicstack-128-exact-opt-prior-6T.yaml \
            -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-stackedu-flat.yaml \
            -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -g ee28fc9c \
            -g 914e1003 \
            -g 2acff647 \
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
            --dashboard regmixer \
            --dashboard mixing-paper \
            --constrain-objective \
            --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
            --requested-tokens 1_000_000_000_000 \
            --repetition-factor $R \
            --drop-metrics ultrachat_masked_ppl \
            --drop-metrics wildchat_masked_ppl \
            --drop-metrics qasper_yesno:rc::olmes \
            --drop-metrics sciriff_yesno:rc::olmes \
            --drop-metrics lab_bench_dbqa \
            --drop-metrics lab_bench_protocolqa \
            --drop-metrics medqa_en:rc::none \
            --train-split 20 \
            --train-split 5 \
            --train-split 15 \
            --train-split 25 \
            --patched

    done
done 

'


# try with strong manual prior 
: 'for SEED in 1 2
do 
    rmc-eval fit -c src/regmixer/config/superswarm-with-strong-manual-prior.yaml \
        -c src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-remove-algebraicstack-128-exact-opt-prior-6T.yaml \
        -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-stackedu-flat.yaml \
        -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g ee28fc9c \
        -g 914e1003 \
        -g 2acff647 \
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
        --dashboard regmixer \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
        --requested-tokens 1_000_000_000_000 \
        --repetition-factor 4 \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 123 \
        --train-split 5 \
        --train-split 64 \
        --train-split 64 \
        --patched
done 
'


: 'for SEED in 0 1 2
do 
    rmc-eval fit -c src/regmixer/config/superswarm-with-strong-manual-prior.yaml \
        -c src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-remove-algebraicstack-128-exact-opt-prior-6T.yaml \
        -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-stackedu-flat.yaml \
        -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g ee28fc9c \
        -g 914e1003 \
        -g 2acff647 \
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
        --dashboard regmixer \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
        --requested-tokens 1_000_000_000_000 \
        --repetition-factor 4 \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 59 \
        --train-split 5 \
        --train-split 32 \
        --train-split 32 \
        --patched
done '


for SEED in 0 1 2
do 
    rmc-eval fit -c src/regmixer/config/superswarm-with-strong-manual-prior.yaml \
        -c src/regmixer/config/for_paper/superswarm/backfill-5xC-30m-superswarm-remove-algebraicstack-128-exact-opt-prior-6T.yaml \
        -c src/regmixer/config/for_paper/backfill-5xC-30m-dclm-stackedu-flat.yaml \
        -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
        -g ee28fc9c \
        -g 914e1003 \
        -g 2acff647 \
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
        --dashboard regmixer \
        --dashboard mixing-paper \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/token_counts_for_paper_superswarm/superswarm_requested_vs_available_tokens.yaml \
        --requested-tokens 1_000_000_000_000 \
        --repetition-factor 4 \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 20 \
        --train-split 5 \
        --train-split 15 \
        --train-split 25 \
        --patched
done 