#!/bin/bash

: 'rmc-internal train \
    -t "dolma2" \
    -c "ai2/ceres" \
    -w "ai2/dolma2" \
    -b "ai2/oe-base" \
    -N 1 \
    -g 1 \
    -i "olmo_1m" \
    -m 165_915_200 \
    -l 2048 \
    -D uint32 \
    -S 42 \
    -p high \
    --device-batch-size 1 \
    -n "test-run-1M" \
    -d "5xC 1M DCLM proposed mix (rep=4) with d=6 domains, n=35 samples, seed=1" \
    -s "for_paper/cost_ablation/top_6_domains_35_samples_seed_1.yaml"
'

: 'BASE_DIR="src/regmixer/internal/config"
YAML_SUBDIR="for_paper/size_ablation"

for yaml in "${BASE_DIR}/${YAML_SUBDIR}"/*.yaml; do
  name="$(basename "$yaml" .yaml)"

  echo "Launching 1M run for config: $name"

  rmc-internal train \
    -t "dolma2" \
    -c "ai2/jupiter" \
    -w "ai2/dolma2" \
    -b "ai2/oe-base" \
    -N 1 \
    -g 1 \
    -i "olmo_1m" \
    -m 165_915_200 \
    -l 2048 \
    -D uint32 \
    -S 42 \
    -p normal \
    --device-batch-size 1 \
    -n "${name}-1m" \
    -d "5xC 1M version of ${name}" \
    -s "${YAML_SUBDIR}/${name}.yaml"
done
'


BASE_DIR="src/regmixer/internal/config"
YAML_SUBDIR="for_paper/size_ablation_backfill"

for yaml in "${BASE_DIR}/${YAML_SUBDIR}"/*.yaml; do
  name="$(basename "$yaml" .yaml)"

  echo "Launching 1M run for config: $name"

  rmc-internal train \
    -t "dolma2" \
    -c "ai2/jupiter" \
    -w "ai2/dolma2" \
    -b "ai2/oe-base" \
    -N 1 \
    -g 1 \
    -i "olmo_1m" \
    -m 165_915_200 \
    -l 2048 \
    -D uint32 \
    -S 42 \
    -p low \
    --device-batch-size 1 \
    -n "${name}-1m" \
    -d "5xC 1M version of ${name}" \
    -s "${YAML_SUBDIR}/${name}.yaml"
done