#!/bin/bash



#first, we do experiments selecting a random subset, not a neighborhood


: 'for SEED in 0 1 2
do 
    for SIZE in 25 50
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
            --train-split $SIZE
    done
done 
'

: 'for SEED in 0 1 2
do 
    for SIZE in 75
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
            --train-split $SIZE
    done
done 
'

# now actually fit the neighborhood models.
: 'for SIZE in 25 50
do 
    # centering around the best mix
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
        --train-split $SIZE \
        --neighborhood dclm-datadelve-5xC-30m-62e7dc06-0058

    # centering around the valid best mix 
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
        --train-split $SIZE \
        --neighborhood dclm-datadelve-5xC-30m-62e7dc06-0038
done 
'

: 'for SIZE in 75
do 
    # centering around the best mix
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
        --train-split $SIZE \
        --neighborhood dclm-datadelve-5xC-30m-62e7dc06-0058

    # centering around the valid best mix 
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
        --train-split $SIZE \
        --neighborhood dclm-datadelve-5xC-30m-62e7dc06-0038
done 
'

for SIZE in 25 #50 75
do 
    # centering around the best mix
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
        --select-top-k-runs $SIZE

done 


