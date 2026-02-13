#!/bin/bash


# hypothesis:  how does fit change with amount of data and dimmension?
for size in 25 50 75 100 #25 50 75 100 125
do 
    for seed in 0 1 2 
    do 
        rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -g 62e7dc06 \
            -G pretraining_tasks_for_paper \
            -a 1 \
            -S 100_000 \
            -s 1 \
            --opt-avg-metric \
            --seed $seed \
            --regression-type autoscale \
            --dashboard mixing-paper \
            --constrain-objective \
            --repetition-factor 4 \
            --manual-token-constraint-path src/regmixer/eval/cost_ablation_1B_5xC_dclm_requested_vs_available_tokens.yaml \
            --requested-tokens 6_000_000_000_000 \
            --n-test 10 \
            --fit-only \
            --train-split $size
    done 
done



for size in 19 38 57 78 95 114 #19 #38 57 76 95 can scale to 129
do 
    for seed in 0 1 2 
    do 
        rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -c src/regmixer/config/for_paper/cost_ablations/backfill-5xC-30m-dclm-top-18-domains.yaml \
            -g 62e7dc06 \
            -g ee3bf7f7 \
            -G pretraining_tasks_for_paper \
            -a 1 \
            -S 100_000 \
            -s 1 \
            --opt-avg-metric \
            --seed $seed \
            --regression-type autoscale \
            --dashboard mixing-paper \
            --support-domains art_and_design \
            --support-domains crime_and_law \
            --support-domains education_and_jobs \
            --support-domains electronics_and_hardware \
            --support-domains entertainment \
            --support-domains finance_and_business \
            --support-domains games \
            --support-domains health \
            --support-domains literature \
            --support-domains politics \
            --support-domains religion \
            --support-domains science_math_and_technology \
            --support-domains social_life \
            --support-domains software \
            --support-domains software_development \
            --support-domains sports_and_fitness \
            --support-domains transportation \
            --support-domains travel_and_tourism \
            --constrain-objective \
            --repetition-factor 4 \
            --manual-token-constraint-path src/regmixer/eval/cost_ablation_1B_5xC_dclm_requested_vs_available_tokens.yaml \
            --requested-tokens 6_000_000_000_000 \
            --n-test 10 \
            --fit-only \
            --train-split $size
    done
done

# NEED TO DO THE LAST 78
for size in 13 26 39 52 65 78 91 104 117 #91 104 117 130  # 78 #26 39 52 65 can scale up to 130
do 
    for seed in 0 1 2
    do
        rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -c src/regmixer/config/for_paper/cost_ablations/backfill-5xC-30m-dclm-top-12-domains.yaml \
            -c src/regmixer/config/for_paper/cost_ablations/backfill-5xC-30m-dclm-top-18-domains.yaml \
            -g 62e7dc06 \
            -g 50f03954 \
            -g ee3bf7f7 \
            -G pretraining_tasks_for_paper \
            -a 1 \
            -S 100_000 \
            -s 1 \
            --opt-avg-metric \
            --seed $seed \
            --regression-type autoscale \
            --dashboard mixing-paper \
            --support-domains crime_and_law \
            --support-domains education_and_jobs \
            --support-domains entertainment \
            --support-domains finance_and_business \
            --support-domains games \
            --support-domains health \
            --support-domains literature \
            --support-domains politics \
            --support-domains religion \
            --support-domains science_math_and_technology \
            --support-domains software \
            --support-domains software_development \
            --constrain-objective \
            --repetition-factor 4 \
            --manual-token-constraint-path src/regmixer/eval/cost_ablation_1B_5xC_dclm_requested_vs_available_tokens.yaml \
            --requested-tokens 6_000_000_000_000 \
            --fit-only \
            --n-test 10 \
            --train-split $size
        done
done 


for size in 7 14 21 28 35 42 49 56 63 70 77 84 91 98 105 112 #42 49 56 63 70 77 84 91 98 105 112 119 #14 21 28 35 can scale up to 125
do 
    for seed in 0 1 2 
    do 
        rmc-eval fit -c src/regmixer/config/dclm-datadelve-5xC-30m-dolma2tok.yaml \
            -c src/regmixer/config/for_paper/cost_ablations/backfill-5xC-30m-dclm-top-6-domains.yaml \
            -c src/regmixer/config/for_paper/cost_ablations/backfill-5xC-30m-dclm-top-12-domains.yaml \
            -c src/regmixer/config/for_paper/cost_ablations/backfill-5xC-30m-dclm-top-18-domains.yaml \
            -g 62e7dc06 \
            -g 028dcd8d \
            -g 50f03954 \
            -g ee3bf7f7 \
            -G pretraining_tasks_for_paper \
            -a 1 \
            -S 100_000 \
            -s 1 \
            --opt-avg-metric \
            --seed $seed \
            --regression-type autoscale \
            --dashboard mixing-paper \
            --support-domains entertainment \
            --support-domains finance_and_business \
            --support-domains games \
            --support-domains health \
            --support-domains politics \
            --support-domains science_math_and_technology \
            --constrain-objective \
            --repetition-factor 4 \
            --manual-token-constraint-path src/regmixer/eval/cost_ablation_1B_5xC_dclm_requested_vs_available_tokens.yaml \
            --requested-tokens 6_000_000_000_000 \
            --train-split $size \
            --fit-only \
            --n-test 10
    done 
done
