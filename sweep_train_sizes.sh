#!/bin/bash

# This script is used to sweep the training sizes of the model

: 'for SEED in 0 1 2 3 4
do 
    for FRAC in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        echo "Training size: $FRAC"
        echo "Seed: $SEED"
        rmc-eval fit -c src/regmixer/config/dclm-datadelve-1xC-30m.yaml -g 0cd83153 -A mmlu_bpb -a 1 -S 100_000 --n-test 50 -t $FRAC --seed $SEED -s 1 
        rmc-eval fit -c src/regmixer/config/dclm-datadelve-format-1xC-30m.yaml -g dbb401a3 -A mmlu_bpb -a 1 -S 100_000 --n-test 50 -t $FRAC --seed $SEED -s 1
    done
done '


: 'for SEED in 0 1 2 3 4
do 
    echo "Seed: $SEED"
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 206f164f -G mmlu_bpb_new -a 1 -S 100_000 -s 1 --opt-avg-metric --n-test 10 --seed $SEED --regression-type log_linear
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 206f164f -G mmlu_bpb_new -a 1 -S 100_000 -s 1 --n-test 10 --seed $SEED
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 206f164f -A mmlu_bpb_new -a 1 -S 100_000 -s 1 --n-test 10 --seed $SEED
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 206f164f -A arc_easy_new -a 1 -S 100_000 -s 1 --n-test 10 --seed $SEED

done'


: 'for SEED in 0 #1 2 3 4
do 
    echo "Seed: $SEED"
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok-subset.yaml -g 4318c7a9 -G mmlu_bpb_new -a 1 -S 100_000 -s 1 --opt-avg-metric --seed $SEED --regression-type log_linear
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok-subset.yaml -g 4318c7a9 -G mmlu_bpb_new -a 1 -S 100_000 -s 1 --seed $SEED
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok-subset.yaml -g 4318c7a9 -A mmlu_bpb_new -a 1 -S 100_000 -s 1 --seed $SEED
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 206f164f -A arc_easy_new -a 1 -S 100_000 -s 1 --seed $SEED  #--regression-type log_linear
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 206f164f -G all_bpb -a 1 -S 100_000 -s 1 --seed $SEED --opt-avg-metric  --regression-type log_linear
    # rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok-subset.yaml -g 4318c7a9 -A c4 -a 1 -S 100_000 -s 1 --seed $SEED --n-test 10 --regression-type log_linear
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok-subset.yaml -g 4318c7a9 -A winogrande -a 1 -S 100_000 -s 1 --n-test 10 --seed $SEED

done' 



: 'for SEED in 0 1 2 3 4
do 
    echo "Seed: $SEED"
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-1xC-30m.yaml -g 0cd83153 -A mmlu_bpb -a 1 -S 100_000 --seed $SEED 
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-format-1xC-30m.yaml -g dbb401a3 -A mmlu_bpb -a 1 -S 100_000  --seed $SEED 
done'


: 'for SEED in 0 1 2
do
    for SPLIT in 0.1 #0.4 0.6 0.8 
    do
        echo "Seed: $SEED"
        echo "Split: $SPLIT"

        #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed $SEED -t $SPLIT
        #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --seed $SEED -t $SPLIT 
        #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -A all_bpb -a 1 -S 100_000 -s 1 --seed $SEED -t $SPLIT 
        rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed $SEED --regression-type log_linear -t $SPLIT
        #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed $SEED --regression-type search --proposer-type search -N #-t $SPLIT
    done
done'

# SPARSE 
#rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 
#rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --seed 0 
#rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -A all_bpb -a 1 -S 100_000 -s 1 --seed 0 
#rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear 
#rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --temperature 1.0

: 'for SEED in 0 1 2 
do
    for SPLIT in 0.1 #0.4 0.6 0.8
    do
        rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed $SEED --regression-type search --proposer-type search -t $SPLIT
    done 
done'

: 'for SPLIT in 0.1 # 0.6 0.8
do
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --seed 0 --opt-avg-metric --regression-type log_linear --neighborhood dclm-datadelve-5xC-30m-62e7dc06-0120 -t $SPLIT
done'

# dense swarm
#rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0
#rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -G all_bpb -a 1 -S 100_000 -s 1 --seed 0 
#rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -A all_bpb -a 1 -S 100_000 -s 1 --seed 0 
#rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear 
#rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type search --proposer-type search 
#rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --temperature 1.0


# dense swarm
: 'for SEED in 0
do
    rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -G all_bpb -a 1 -S 100_000 -s 1 --n-test 10 --opt-avg-metric --seed 0
    rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -G all_bpb -a 1 -S 100_000 -s 1 --n-test 10 --opt-avg-metric --seed $SEED --regression-type log_linear 

done '


# unconstrained universe 
#rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --constrain-swarm
# rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --constrain-objective
#rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear  --constrain-swarm
#rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear  --constrain-objective
#rmc-eval fit -c src/regmixer/config/dclm-larger-datadelve-5xC-30m.yaml -g 0b5a9356 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear  --constrain-objective --constrain-swarm


# aggregated objs
#rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 62e7dc06 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights aggregated_minerva_codex --constrain-objective


# OLMO2 MIX 
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear 
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type search --proposer-type search 
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear  --neighborhood olmo2-5xC-30m-ef272e64-0051 -t 0.4 

#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights aggregated_minerva_codex
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights aggregated_minerva_codex --temperature 0.1 --constrain-objective --final-cookbook-path /home/mayee/re/year6/ai2/olmo-cookbook/src/cookbook/recipes/train-1b-v2-5xC-olmo2-mix-natural.yaml

# using both the wiki-upsampled mix and the regular mix 
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -g 25d7ec63 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights aggregated_minerva_codex

# using only the wiki-upsampled mix
# rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g 25d7ec63 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights aggregated_minerva_codex --temperature 0.1


# OLMO2 mix with constraints 
# rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --constrain-swarm --constrain-objective --final-cookbook-path /home/mayee/re/year6/ai2/olmo-cookbook/src/cookbook/recipes/train-1b-v2-5xC-olmo2-mix-natural.yaml
# rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --constrain-objective --final-cookbook-path /home/mayee/re/year6/ai2/olmo-cookbook/src/cookbook/recipes/train-1b-v2-5xC-olmo2-mix-natural.yaml



# fit with offline tasks too 
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb_with_offline -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear 
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb_with_offline -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights aggregated_minerva_codex_mtmbpp_skills
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb_with_offline -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights correlation_weighting
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb_with_offline -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights correlation_weighting --temperature 0.1 --constrain-objective --final-cookbook-path /home/mayee/re/year6/ai2/olmo-cookbook/src/cookbook/recipes/train-1b-v2-5xC-olmo2-mix-natural.yaml

# fit regression on dclm + wiki only
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -c src/regmixer/config/olmo2-5xC-30m-wiki.yaml -c src/regmixer/config/dclm-wiki-5xC-30m.yaml -g ef272e64 -g 25d7ec63 -g 2bfce224 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --keep-sources dclm --keep-sources wikipedia
#rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -c src/regmixer/config/olmo2-5xC-30m-wiki.yaml -c src/regmixer/config/starcoder-wiki-5xC-30m.yaml -g ef272e64 -g 25d7ec63 -g c49377d4 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --keep-sources starcoder --keep-sources wikipedia



# optimize dummy
# rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 1_000_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights mbpp_dummy --temperature 0.1 # 1.5589739084243774
# rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 1_000_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights mbpp_dummy # 1.5916509628295898
# rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights minerva_dummy # 1.3882726430892944
# rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m.yaml -g ef272e64 -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights minerva_dummy --temperature 0.1 # 1.3073177337646484


# OLMo 2 DENSE MIX
# rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m-dense.yaml -g 6cf425be -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear 
# rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m-dense.yaml -g 6cf425be -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type search --proposer-type search 
# rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m-dense.yaml -g 6cf425be -G all_bpb -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights aggregated_minerva_codex --temperature 0.1


rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m-dense.yaml -g 6cf425be -G all_bpb_with_offline -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights correlation_weighting_dense
rmc-eval fit -c src/regmixer/config/olmo2-5xC-30m-dense.yaml -g 6cf425be -G all_bpb_with_offline -a 1 -S 100_000 -s 1 --opt-avg-metric --seed 0 --regression-type log_linear --obj-weights correlation_weighting
