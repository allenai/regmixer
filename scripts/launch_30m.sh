#!/bin/bash



: 'BASE_DIR="src/regmixer/internal/config"
YAML_SUBDIR="for_paper/size_ablation_backfill"

for yaml in "${BASE_DIR}/${YAML_SUBDIR}"/*.yaml; do
  name="$(basename "$yaml" .yaml)"

  echo "Launching 30M run for config: $name"

  rmc-internal train \
    -t "dolma2" \
    -c "ai2/jupiter" \
    -w "ai2/dolma2" \
    -b "ai2/oe-base" \
    -N 1 \
    -g 1 \
    -i "olmo_30m" \
    -m 2_910_233_600 \
    -l 2048 \
    -D uint32 \
    -S 42 \
    -p low \
    --device-batch-size 32 \
    -n "${name}-30m" \
    -d "5xC 30M version of ${name}" \
    -s "${YAML_SUBDIR}/${name}.yaml"
done'


BASE_DIR="src/regmixer/internal/config"
YAML_SUBDIR="for_paper/size_ablation"

for yaml in "${BASE_DIR}/${YAML_SUBDIR}"/*.yaml; do
  name="$(basename "$yaml" .yaml)"

  echo "Launching 30M run for config: $name"

  rmc-internal train \
    -t "dolma2" \
    -c "ai2/titan" \
    -w "ai2/dolma2" \
    -b "ai2/oe-base" \
    -N 1 \
    -g 1 \
    -i "olmo_30m" \
    -m 2_910_233_600 \
    -l 2048 \
    -D uint32 \
    -S 42 \
    -p high \
    --device-batch-size 32 \
    -n "${name}-30m-redo" \
    -d "5xC 30M version of ${name}" \
    -s "${YAML_SUBDIR}/${name}.yaml"
done