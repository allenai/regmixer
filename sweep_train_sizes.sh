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


for SEED in 0 #1 2 3 4
do 
    echo "Seed: $SEED"
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok-subset.yaml -g 4318c7a9 -G mmlu_bpb_new -a 1 -S 100_000 -s 1 --opt-avg-metric --seed $SEED --regression-type log_linear
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok-subset.yaml -g 4318c7a9 -G mmlu_bpb_new -a 1 -S 100_000 -s 1 --seed $SEED
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok-subset.yaml -g 4318c7a9 -A mmlu_bpb_new -a 1 -S 100_000 -s 1 --seed $SEED
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 206f164f -A arc_easy_new -a 1 -S 100_000 -s 1 --seed $SEED  #--regression-type log_linear
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml -g 206f164f -G all_bpb -a 1 -S 100_000 -s 1 --seed $SEED --opt-avg-metric  --regression-type log_linear
    # rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok-subset.yaml -g 4318c7a9 -A c4 -a 1 -S 100_000 -s 1 --seed $SEED --n-test 10 --regression-type log_linear
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok-subset.yaml -g 4318c7a9 -A winogrande -a 1 -S 100_000 -s 1 --n-test 10 --seed $SEED

done



: 'for SEED in 0 1 2 3 4
do 
    echo "Seed: $SEED"
    rmc-eval fit -c src/regmixer/config/dclm-datadelve-1xC-30m.yaml -g 0cd83153 -A mmlu_bpb -a 1 -S 100_000 --seed $SEED 
    #rmc-eval fit -c src/regmixer/config/dclm-datadelve-format-1xC-30m.yaml -g dbb401a3 -A mmlu_bpb -a 1 -S 100_000  --seed $SEED 
done'