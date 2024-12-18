```bash
rmc-internal train \
    -n avg_validation_loss_regmixer_optimal_1B \
    -t gpt_neox \
    -c ai2/jupiter-cirrascale-2 \
    -w ai2/dolma2 \
    -b ai2/oe-data \
    -m 52_000_000_000 \
    -l 2048 \
    -p high \
    -g 8 \
    -N 4 \
    -d uint16 \
    -r 5.0 \
    -s avg_validation_loss_alpha_50_constrained.yaml
```
