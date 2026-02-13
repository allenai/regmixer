#!/bin/bash
: 'rmc-internal train -t "dolma2" -c "ai2/jupiter" -w "ai2/dolma2" -b "ai2/oe-base" -N 1 -g 1 -i "olmo_30m" -m 2_910_233_600 -l 2048 -D uint32 -S 42 -p high --device-batch-size 32 -n "30m-5xC-dclm-aggregate-exact-6T" -d "5xC 30M DCLM proposed mix (rep=4, 6T) with aggregate" -s "for_paper/granularity/30m-5xC-dclm-aggregate-exact-6T.yaml"
rmc-internal train -t "dolma2" -c "ai2/jupiter" -w "ai2/dolma2" -b "ai2/oe-base" -N 1 -g 1 -i "olmo_30m" -m 2_910_233_600 -l 2048 -D uint32 -S 42 -p high --device-batch-size 32 -n "30m-5xC-dclm-per-task-exact-6T" -d "5xC 30M DCLM proposed mix (rep=4, 6T) with per-task" -s "for_paper/granularity/30m-5xC-dclm-per-task-exact-6T.yaml"
rmc-internal train -t "dolma2" -c "ai2/jupiter" -w "ai2/dolma2" -b "ai2/oe-base" -N 1 -g 1 -i "olmo_30m" -m 2_910_233_600 -l 2048 -D uint32 -S 42 -p high --device-batch-size 32 -n "30m-5xC-dclm-per-family-exact-6T" -d "5xC 30M DCLM proposed mix (rep=4, 6T) with per-family" -s "for_paper/granularity/30m-5xC-dclm-per-family-exact-6T.yaml"
'



rmc-internal train -t "dolma2" -c "ai2/jupiter" -w "ai2/dolma2" -b "ai2/oe-base" -N 1 -g 1 -i "olmo_30m" -m 2_910_233_600 -l 2048 -D uint32 -S 42 -p high --device-batch-size 32 -n "30m-5xC-dclm-per-task-exact-6T-larger-sample" -d "5xC 30M DCLM proposed mix (rep=4, 6T) with per-task" -s "for_paper/granularity/30m-5xC-dclm-per-task-exact-6T-larger-sample.yaml"
rmc-internal train -t "dolma2" -c "ai2/jupiter" -w "ai2/dolma2" -b "ai2/oe-base" -N 1 -g 1 -i "olmo_30m" -m 2_910_233_600 -l 2048 -D uint32 -S 42 -p high --device-batch-size 32 -n "30m-5xC-dclm-aggregate-exact-6T-larger-sample" -d "5xC 30M DCLM proposed mix (rep=4, 6T) with aggregate" -s "for_paper/granularity/30m-5xC-dclm-aggregate-exact-6T-larger-sample.yaml"
rmc-internal train -t "dolma2" -c "ai2/jupiter" -w "ai2/dolma2" -b "ai2/oe-base" -N 1 -g 1 -i "olmo_30m" -m 2_910_233_600 -l 2048 -D uint32 -S 42 -p high --device-batch-size 32 -n "30m-5xC-dclm-per-family-exact-6T-larger-sample" -d "5xC 30M DCLM proposed mix (rep=4, 6T) with per-family" -s "for_paper/granularity/30m-5xC-dclm-per-family-exact-6T-larger-sample.yaml"
