#!/bin/bash



# the right one 
for R in 4
do
    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G qa_tasks \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R 

    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G math_tasks \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R 
    '

    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_1T_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R

    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_1T_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R \
        --obj-weights code_tasks_new_weights'

    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R

    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R \
        --obj-weights code_tasks_new_weights
'


    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor 5 \
        --obj-weights code_tasks_new_weights

    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor 6 \
        --obj-weights code_tasks_new_weights
'


    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R \
        --obj-weights code_tasks_new_weights_extreme'

   : ' rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_1T_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R \
        --obj-weights code_tasks_new_weights_extreme'

    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer

    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --obj-weights code_tasks_new_weights'


    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G code_tasks_new \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --obj-weights code_tasks_new_weights_extreme
'

   : ' rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_1T_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 256

    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_1T_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
'


: '    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 256

    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 256
'

: '    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \

    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
'
    # 512 exact
: '    rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest_FIXED_PES2O.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
'

    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --proposer-type exact \
        --kl-reg 0.05 \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest_FIXED_PES2O.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 256'

    : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest_FIXED_PES2O.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
        --train-split 256
    '
    : ' rmc-eval fit -c src/regmixer/config/superswarm.yaml \
        -g ee28fc9c \
        -G pretraining_tasks_for_paper \
        -a 1 \
        -S 100_000 \
        -s 1 \
        --opt-avg-metric \
        --seed 0 \
        --regression-type log_linear \
        --dashboard regmixer \
        --constrain-objective \
        --manual-token-constraint-path src/regmixer/eval/superswarm_requested_vs_available_tokens_newest_FIXED_PES2O.yaml \
        --repetition-factor $R \
        --drop-metrics ultrachat_masked_ppl \
        --drop-metrics wildchat_masked_ppl \
        --drop-metrics qasper_yesno:rc::olmes \
        --drop-metrics sciriff_yesno:rc::olmes \
        --drop-metrics lab_bench_dbqa \
        --drop-metrics lab_bench_protocolqa \
        --drop-metrics medqa_en:rc::none \
'



    # 256 seed 0 6T exact with FIXED TOKEN POOLS
: '    for SEED in 0 1 2 
    do 
        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -g ee28fc9c \
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
            --train-split 256
    done 
'

    : 'for SEED in 0 1 2 
    do 
        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -g ee28fc9c \
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
            --train-split 256
    done'


    : 'for SEED in 0 
    do 
        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -g ee28fc9c \
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
            --train-split 128
    done
'

    for SEED in 0 
    do 
        : 'rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -g ee28fc9c \
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
            --train-split 128
            '
        rmc-eval fit -c src/regmixer/config/superswarm.yaml \
            -g ee28fc9c \
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
            --train-split 65

    done

done 


