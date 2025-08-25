#!/bin/bash

: 'for i in $(seq -f "%04g" 0 63); do
  if [ "$i" == "0049" ]; then
    step="step22200-hf"
  else
    step="step22100-hf"
  fi

  echo "Evaluating checkpoint $i with $step..."

  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/olmo2-5xC-30m-ef272e64-${i}/${step}" \
    --tasks basic_skills:rc::olmes --tasks mt_mbpp \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'





: 'experiments=(
 'olmo2-aggregate-minerva-codex-16e92c9a'
 'olmo2-log-linear-4f3520ff'
 'olmo2-with-offline-aggregate-code-math-30d195d0'
 'olmo-cookbook-core-v2-1bv2-5xC-olmo2-mix-natural-fcb5f8e2'
)'

: 'experiments=(
  'olmo2-with-offline-f45f51fa'
  'olmo2-with-offline-correlation-weighting-cc04b2b8'
  'olmo2-1b-log-linear-dense-21d5fcf6'
)'


# Loop over each experiment ID
: 'for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks basic_skills:rc::olmes --tasks mt_mbpp \
    --priority high \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'




: 'for i in $(seq -f "%04g" 0 63); do
  step="step22100-hf"
  echo "Evaluating checkpoint $i with $step..."
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/olmo2-5xC-30m-dense-6cf425be-${i}/${step}" \
    --tasks basic_skills:rc::olmes --tasks mt_mbpp \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'





: 'for i in $(seq -f "%04g" 0 127); do
  echo "Evaluating checkpoint $i with $step..."
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/dclm-datadelve-5xC-30m-62e7dc06-${i}/step22100-hf" \
    --tasks basic_skills:rc::olmes --tasks mt_mbpp \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'




: 'experiments=(
  'dclm-offline-correlation-weighting-from-sparse-olmo2-77341634'
  'dclm-offline-correlation-weighting-from-dense-olmo2-b79f5bcf'
  'dclm-offline-correlation-weighting-from-dclm-b5933d40'
  'dclm-log-linear-aggregate-minerva-codex-constrain-obj-63331f15'
)
for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks basic_skills:rc::olmes --tasks mt_mbpp \
    --priority high \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'





: 'experiments=(
  "olmo2-1b-aggregate-minerva-codex-diverse-constrain-obj-25a78cf2"
  "olmo2-aggregate-minerva-codex-diverse-6b5e6618"
)
for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks basic_skills:rc::olmes --tasks mt_mbpp \
    --priority high \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'








: 'for i in $(seq -f "%04g" 0 127); do
  step="step22100-hf"
  echo "Evaluating checkpoint $i with $step..."
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/dclm-datadelve-5xC-30m-62e7dc06-${i}/step22100-hf" \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done


for i in $(seq -f "%04g" 0 127); do
  step="step22100-hf"
  echo "Evaluating checkpoint $i with $step..."
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/dclm-datadelve-5xC-30m-62e7dc06-${i}/step22100-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'




: 'experiments=(
  'regmixer-dclm-log-linear-offline-extended-constrain-obj-0e5a89aa-0000'
  'regmixer-dclm-log-linear-offline-extended-correlation-weighted-constrain-obj-982e6847-0000'
  'regmixer-dclm-log-linear-aggregate-minerva-codex-constrain-obj-492be78c-0000'
  'regmixer-all_bpb_log_linear_constrain_obj_on_large-ef3c8b47-0000'
  'regmixer-all_bpb_log_linear_on_large-f6a9f12c-0000'
)'

: 'experiments=(
  'regmixer-natural-larger-sample-11909fd9-0000'
)'

: 'experiments=(
  'regmixer-natural-smaller-sample-922bf462-0000'
)'

: 'experiments=(
  'regmixer-dclm-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-30856821-0000'
  'regmixer-dclm-log-linear-full-eval-constrain-obj-pareto-with-true-scores-0d2c2fff-0000'
  'regmixer-dclm-log-linear-full-eval-correlation-weighting-constrain-obj-pareto-with-predicted-scores-f3adef2b-0000'
  'regmixer-dclm-log-linear-full-eval-correlation-weighting-constrain-obj-pareto-with-true-scores-987bf9cc-0000'
  'regmixer-dclm-offline-correlation-weighting-from-dense-olmo2-39a7ba4d-0000'
)'

: 'experiments=(
  #'regmixer-superswarm-log-linear-cb34591f-0000'
  'regmixer-superswarm-log-linear-underfit-6bfe47f7-0000'
  #'regmixer-superswarm-natural-b5dbea02-0000'
)'
  

: 'for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step22100-hf" \
    --tasks basic_skills:rc::olmes \
    --tasks mt_mbpp \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'

: 'for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step22080-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'


  

: 'experiments=(
    '1b-5xC-superswarm-log-linear-cf0dd59f'
    '1b-5xC-superswarm-log-linear-underfit-1b2af97c'
    '1b-5xC-superswarm-natural-4d1a1440'
)'

: 'experiments=(
    'dclm-1b-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-173bec67'
    'dclm-1b-log-linear-full-eval-constrain-obj-pareto-with-true-scores-4557a88b'
    'dclm-1b-log-linear-full-eval-corr-weighting-constrain-obj-pareto-with-predicted-scores-b9e80c0a'
    'dclm-1b-log-linear-full-eval-corr-weighting-constrain-obj-pareto-with-true-scores-e07ba770'
)'

: 'experiments=(
    'dclm-1b-log-linear-extended-eval-constrain-obj-c24744d0'
)'


: 'experiments=(
  'olmo-cookbook-core-v2-1bv2-5xC-dclm-baseline-topic-classified-sample-natural-28f8e9a9'
)'



: 'for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks mt_mbpp \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'

: 'for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'



: 'experiments=(
    'dclm-1b-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-d7eaca81'
    #'dclm-1b-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-357a137e'
)'



: 'for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks mt_mbpp \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'

: 'for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'


: 'experiments=(
  'dclm-offline-correlation-weighting-from-dense-olmo2-b79f5bcf'
)


for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done

for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done
'

: 'experiments=(
  #'dclm-offline-correlation-weighting-from-sparse-olmo2-77341634'
  'dclm-offline-correlation-weighting-from-dense-olmo2-b79f5bcf'
  #'dclm-offline-correlation-weighting-from-dclm-b5933d40'
  #'dclm-log-linear-aggregate-minerva-codex-constrain-obj-63331f15'
)
for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks basic_skills:rc::olmes --tasks mt_mbpp \
    --priority high \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'



: 'experiments=(
  "regmixer-dclm-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-repetition-5-da5e6427-0000"
  "regmixer-dclm-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-repetition-2.5-3fcade6c-0000"
  "regmixer-dclm-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-repetition-3-d42d25ec-0000"
  "regmixer-dclm-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-repetition-3.5-cfad9c20-0000"
  "regmixer-dclm-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-repetition-4-76017bfc-0000"
  "regmixer-dclm-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-repetition-4.5-98222e55-0000"
  "regmixer-dclm-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-repetition-2-005c333c-0000"
)

for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step22100-hf" \
    --tasks mt_mbpp \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done

for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/${exp_id}/step22100-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'







# List of runs with step22100
: 'runs_with_22100=(
  0001 0003 0004 0005 0006 0007 0009 0012 0013 0014 0018 0019 0021 0022 0023 0024
  0026 0028 0029 0031 0034 0036 0037 0039 0040 0042 0049 0056 0059 0062 0064 0065
  0066 0067 0068 0069 0072 0075 0078 0081 0082 0085 0086 0087 0089 0090 0091 0093
  0094 0095 0096 0097 0098 0100 0102 0103 0107 0108 0109 0110 0112 0115 0116 0119
  0120 0123 0124 0125 0127
)

# List of runs with step22200
runs_with_22200=(
  0000 0002 0008 0010 0011 0015 0016 0017 0020 0025 0027 0030 0032 0033 0035 0038
  0041 0044 0046 0048 0054 0055 0057 0060 0061 0063 0070 0071 0073 0074 0076 0077
  0079 0080 0083 0084 0088 0092 0099 0101 0104 0105 0106 0111 0113 0114 0117 0118
  0121 0122 0126
)

# ---------- FIRST TASK GROUP ----------


for exp_id in "${runs_with_22100[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-e250076d-${exp_id}/step22100-hf" \
    --tasks mt_mbpp \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done

for exp_id in "${runs_with_22200[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-e250076d-${exp_id}/step22200-hf" \
    --tasks mt_mbpp \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done

# ---------- SECOND TASK GROUP ----------

for exp_id in "${runs_with_22100[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-e250076d-${exp_id}/step22100-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done

for exp_id in "${runs_with_22200[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-e250076d-${exp_id}/step22200-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done



# Runs with step22100
runs_with_22100=(0002 0003 0004 0005 0006 0007)

# Runs with step22200
runs_with_22200=(0000 0001)

# ---------- FIRST TASK GROUP ----------

for exp_id in "${runs_with_22100[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-a0e3840c-${exp_id}/step22100-hf" \
    --tasks mt_mbpp \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done

for exp_id in "${runs_with_22200[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-a0e3840c-${exp_id}/step22200-hf" \
    --tasks mt_mbpp \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done

# ---------- SECOND TASK GROUP ----------

for exp_id in "${runs_with_22100[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-a0e3840c-${exp_id}/step22100-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done

for exp_id in "${runs_with_22200[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-a0e3840c-${exp_id}/step22200-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'






# Define ignore list
#ignore_set=("0502" "0501" "0479" "0478" "0451" "0406" "0375" "0301" "0296")


: 'for i in $(seq -f "%04g" 2 511); do
    # Skip if in ignore list
    if [[ " ${ignore_set[@]} " =~ " ${i} " ]]; then
        echo "Skipping part-$i (in ignore list)"
        continue
    fi

    echo "Syncing checkpoint part-$i..."

    olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-dclm-conditional-4252db91-$i/step22100-hf" \
    --tasks "*olmo3:dev:1b:vllm" \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2 \
    --partition-size 8 

    olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-dclm-conditional-4252db91-$i/step22100-hf" \
    --tasks "*olmo3:dev:1b:hf" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2

done'


#ignore_set=("0492")

: 'for i in "${ignore_set[@]}"; do
    echo "Syncing checkpoint part-$i..."

    olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-dclm-conditional-4252db91-$i/step22100-hf" \
    --tasks "*olmo3:dev:1b:vllm" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2 \
    --partition-size 8 
done'

    : 'olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-dclm-conditional-4252db91-$i/step22100-hf" \
    --tasks "*olmo3:dev:1b:hf" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2'




: 'experiments=(
    'regmixer-stack-natural-a4e0986e-0000'
)'

: 'experiments=(
  'regmixer-pdf-natural-ded93f25-0000'
)


for exp_id in "${experiments[@]}"; do
    olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/$exp_id/step22100-hf" \
    --tasks "*olmo3:dev:1b:vllm" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2 \

    olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/mayeec/$exp_id/step22100-hf" \
    --tasks "*olmo3:dev:1b:hf" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2 \

    
done '





: 'for i in $(seq -f "%04g" 0 63); do
    echo "Syncing checkpoint part-$i..."
    olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/ai2-tylerm/olmo2-pdfs-datadelve-5xC-30m-augusta-2048-8b10a86d-$i/step22100-hf" \
    --tasks "*olmo3:dev:1b:vllm" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2 \
    --partition-size 8 

    olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/ai2-tylerm/olmo2-pdfs-datadelve-5xC-30m-augusta-2048-8b10a86d-$i/step22100-hf" \
    --tasks "*olmo3:dev:1b:hf" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2

done'



: 'ignore_set=("0028" "0023" "0012")
ignore_set=("0008" "0019")

for i in "${ignore_set[@]}"; do
    echo "Syncing checkpoint part-$i..."

    olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/ai2-tylerm/olmo2-pdfs-datadelve-5xC-30m-augusta-2048-6d2d4c39-$i/step22200-hf" \
    --tasks "*olmo3:dev:1b:vllm" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2 \
    --partition-size 8 

    olmo-cookbook-eval evaluate \
    "/oe-training-default/ai2-llm/checkpoints/ai2-tylerm/olmo2-pdfs-datadelve-5xC-30m-augusta-2048-6d2d4c39-$i/step22200-hf" \
    --tasks "*olmo3:dev:1b:hf" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'


: 'experiments=(
    '1b-5xC-superswarm-pareto-repetition-constraint-4.5-09dc1b5d'
    '1b-5xC-superswarm-repetition-constraint-4-e999ce68'
)'

: 'for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks mt_mbpp \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done

for exp_id in "${experiments[@]}"; do
  olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/${exp_id}/step61000-hf" \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2
done'



# Generate full range and exclude specific numbers
: 'FULL_RANGE=($(seq -f "%04g" 0 127))
EXCLUDED=(0507 0508)
INCLUDED=()

for num in "${FULL_RANGE[@]}"; do
    skip=false
    for exclude in "${EXCLUDED[@]}"; do
        if [[ "$num" == "$exclude" ]]; then
            skip=true
            break
        fi
    done
    if [[ "$skip" == false ]]; then
        INCLUDED+=("$num")
    fi
done

# Function to count running background jobs
count_jobs() {
    jobs -r | wc -l
}

# Process all checkpoints with max 12 concurrent jobs
for i in "${INCLUDED[@]}"; do
    # Wait until we have fewer than 12 jobs running
    while [ $(count_jobs) -ge 12 ]; do
        sleep 5
    done

    echo "Starting evaluation for checkpoint $i..."
    {

      olmo-cookbook-eval evaluate \
      "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-dense-fb40b847-$i/step22100-hf" \
      --tasks "*olmo3:dev:1b:vllm" \
      --priority urgent \
      --cluster ai2/jupiter-cirrascale-2 \
      --num-gpus 1 \
      --model-backend vllm \
      --model-args dtype=bfloat16 \
      --dashboard regmixer \
      --workspace ai2/dolma2 \
      --partition-size 8 

      olmo-cookbook-eval evaluate \
      "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-dense-fb40b847-$i/step22100-hf" \
      --tasks "*olmo3:dev:1b:hf" \
      --priority urgent \
      --cluster ai2/jupiter-cirrascale-2 \
      --num-gpus 1 \
      --model-backend hf \
      --model-args dtype=bfloat16 \
      --dashboard regmixer \
      --workspace ai2/dolma2


      echo "Completed checkpoint $i"
    } &
done

# Wait for all remaining jobs to complete
wait
echo "All evaluation tasks completed!"
'




# Generate full range and exclude specific numbers
: 'FULL_RANGE=($(seq -f "%04g" 0 127))
EXCLUDED=(0507 0508)
INCLUDED=()

for num in "${FULL_RANGE[@]}"; do
    skip=false
    for exclude in "${EXCLUDED[@]}"; do
        if [[ "$num" == "$exclude" ]]; then
            skip=true
            break
        fi
    done
    if [[ "$skip" == false ]]; then
        INCLUDED+=("$num")
    fi
done

# Function to count running background jobs
count_jobs() {
    jobs -r | wc -l
}

# Process all checkpoints with max 12 concurrent jobs
for i in "${INCLUDED[@]}"; do
    # Wait until we have fewer than 12 jobs running
    while [ $(count_jobs) -ge 12 ]; do
        sleep 5
    done

    echo "Starting evaluation for checkpoint $i..."
    {

      olmo-cookbook-eval evaluate \
      "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-dense-fb40b847-$i/step22100-hf" \
      --tasks mt_mbpp \
      --priority urgent \
      --cluster ai2/jupiter-cirrascale-2 \
      --num-gpus 1 \
      --model-backend vllm \
      --model-args dtype=bfloat16 \
      --dashboard regmixer \
      --workspace ai2/dolma2

      echo "Completed checkpoint $i"
    } &
done

# Wait for all remaining jobs to complete
wait
echo "All evaluation tasks completed!"
'



experiments=(    
  '1b-5xC-superswarm-conditional-dclm-log-linear-78403bc2'
  '1b-5xC-superswarm-v1-log-linear-on-v2-f15e08c9'
  '1b-5xC-superswarm-v2-natural-52749eca'
  '1b-5xC-superswarm-v2-conditional-dclm-natural-61de5e1e'

)


: 'experiments=(
    'regmixer-superswarm-v2-conditional-dclm-log-linear-2e6e1ff0-0000'
    'regmixer-superswarm-v1-log-linear-on-v2-fd7cab69-0000'
    'regmixer-superswarm-v2-conditional-dclm-41533660-0000'
    'regmixer-superswarm-v2-natural-4f36a33e-0000'
)'



: 'experiments=(
    #'1b-5xC-superswarm-conditional-dclm-log-linear-constraint-3-4ca6c07f'
    #'1b-5xC-superswarm-conditional-dclm-log-linear-constraint-4-c1930276'
    #'1b-5xC-superswarm-conditional-dclm-log-linear-constraint-5-7e702ff7'
    #'1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-b65c9504'
    #'1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-constraint-3-47953a8c'
    #'1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-constraint-4-2099326f'
    #'1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-constraint-5-0b61d5af'
)'


: 'for exp_id in "${experiments[@]}"; do
    olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step61000-hf" \
    --tasks "*olmo3:dev:1b:vllm" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2 \
    --partition-size 8 


    olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step61000-hf" \
    --tasks "*olmo3:dev:1b:hf" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2

    
done'




experiments=(    
  '1b-5xC-superswarm-conditional-dclm-log-linear-78403bc2'
  '1b-5xC-superswarm-v1-log-linear-on-v2-f15e08c9'
  '1b-5xC-superswarm-v2-natural-52749eca'
  '1b-5xC-superswarm-v2-conditional-dclm-natural-61de5e1e'

)

: 'for exp_id in "${experiments[@]}"; do
    olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step61000-hf" \
    --tasks "sciq:rc::olmo3" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2 
done'




experiments=(
    'regmixer-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-constraint-3-660a6d04-0000'
    'regmixer-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-constraint-4-1b497b25-0000'
    'regmixer-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-constraint-5-00670414-0000'
    'regmixer-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-02a77d23-0000'
)


experiments=(
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-e4d26d38'
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-constraint-3-7df4e2aa'
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-constraint-5-dc4cc209'
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-constraint-fixed-4-549f2d14'
)



: 'for exp_id in "${experiments[@]}"; do
    olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step61000-hf" \
    --tasks "*olmo3:dev:1b:vllm" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2 \
    --partition-size 8 

    olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step61000-hf" \
    --tasks "*olmo3:dev:1b:hf" \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --dashboard regmixer \
    --workspace ai2/dolma2

done
'



: 'FULL_RANGE=($(seq -f "%04g" 0 1))
EXCLUDED=(0507 0508)
INCLUDED=()

for num in "${FULL_RANGE[@]}"; do
    skip=false
    for exclude in "${EXCLUDED[@]}"; do
        if [[ "$num" == "$exclude" ]]; then
            skip=true
            break
        fi
    done
    if [[ "$skip" == false ]]; then
        INCLUDED+=("$num")
    fi
done

# Function to count running background jobs
count_jobs() {
    jobs -r | wc -l
}

# Process all checkpoints with max 12 concurrent jobs
for i in "${INCLUDED[@]}"; do
    # Wait until we have fewer than 12 jobs running
    while [ $(count_jobs) -ge 12 ]; do
        sleep 5
    done

    echo "Starting evaluation for checkpoint $i..."
    {

      olmo-cookbook-eval evaluate \
      "/oe-data-default/ai2-llm/checkpoints/ai2-tylerm/5xC-30m-superswarm-dclm-stackedu-conditional-0cb55cb5-$i/step22100-hf" \
      --tasks "*olmo3:dev:1b:vllm" \
      --priority urgent \
      --cluster ai2/augusta-google-1 \
      --num-gpus 1 \
      --model-backend vllm \
      --model-args dtype=bfloat16 \
      --dashboard regmixer \
      --workspace ai2/dolma2 \
      --partition-size 8 

      olmo-cookbook-eval evaluate \
      "/oe-data-default/ai2-llm/checkpoints/ai2-tylerm/5xC-30m-superswarm-dclm-stackedu-conditional-0cb55cb5-$i/step22100-hf" \
      --tasks "*olmo3:dev:1b:hf" \
      --priority urgent \
      --cluster ai2/augusta-google-1 \
      --num-gpus 1 \
      --model-backend hf \
      --model-args dtype=bfloat16 \
      --dashboard regmixer \
      --workspace ai2/dolma2



      echo "Completed checkpoint $i"
    } &
done

# Wait for all remaining jobs to complete
wait
echo "All evaluation tasks completed!"
'


FULL_RANGE=($(seq -f "%04g" 0 127))
EXCLUDED=()
INCLUDED=()

for num in "${FULL_RANGE[@]}"; do
    skip=false
    for exclude in "${EXCLUDED[@]}"; do
        if [[ "$num" == "$exclude" ]]; then
            skip=true
            break
        fi
    done
    if [[ "$skip" == false ]]; then
        INCLUDED+=("$num")
    fi
done

INCLUDED=("0001" "0002" "0005" "0006" "0007" "0019" "0023" "0028" "0033" "0041" "0057" "0066" "0070" "0072" "0073" "0074" "0077" "0083" "0086" "0088" "0101" "0111" "0113" "0117" "0124" "0127")

# Function to count running background jobs
count_jobs() {
    jobs -r | wc -l
}

: '# Process all checkpoints with max 12 concurrent jobs
for i in "${INCLUDED[@]}"; do
    # Wait until we have fewer than 12 jobs running
    while [ $(count_jobs) -ge 12 ]; do
        sleep 5
    done

    echo "Starting evaluation for checkpoint $i..."
    {

      olmo-cookbook-eval evaluate \
      "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-vigintiles-32b90003-$i/step22200-hf" \
      --tasks "olmo3:dev:1b:main" \
      --priority high \
      --cluster ai2/augusta-google-1 \
      --num-gpus 1 \
      --model-backend vllm \
      --model-args dtype=bfloat16 \
      --dashboard regmixer \
      --workspace ai2/dolma2 \
      --partition-size 8 

      olmo-cookbook-eval evaluate \
      "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-vigintiles-32b90003-$i/step22200-hf" \
      --tasks "olmo3:dev:1b:main:hf" \
      --priority high \
      --cluster ai2/augusta-google-1 \
      --num-gpus 1 \
      --model-backend hf \
      --model-args dtype=bfloat16 \
      --dashboard regmixer \
      --workspace ai2/dolma2

      echo "Completed checkpoint $i"
    } &
done

# Wait for all remaining jobs to complete
wait
echo "All evaluation tasks completed!"
'



FULL_RANGE=($(seq -f "%04g" 0 127))
EXCLUDED=()
INCLUDED=()

for num in "${FULL_RANGE[@]}"; do
    skip=false
    for exclude in "${EXCLUDED[@]}"; do
        if [[ "$num" == "$exclude" ]]; then
            skip=true
            break
        fi
    done
    if [[ "$skip" == false ]]; then
        INCLUDED+=("$num")
    fi
done

# Process all checkpoints with max 12 concurrent jobs
for i in "${INCLUDED[@]}"; do
    # Wait until we have fewer than 12 jobs running
    while [ $(count_jobs) -ge 12 ]; do
        sleep 5
    done

    echo "Starting evaluation for checkpoint $i..."
    {

      olmo-cookbook-eval evaluate \
      "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-all-dressed-topics-95d5a161-$i/step22200-hf" \
      --tasks "olmo3:dev:1b:main" \
      --priority high \
      --cluster ai2/jupiter-cirrascale-2 \
      --num-gpus 1 \
      --model-backend vllm \
      --model-args dtype=bfloat16 \
      --dashboard regmixer \
      --workspace ai2/dolma2 \
      --partition-size 8 

      olmo-cookbook-eval evaluate \
      "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-all-dressed-topics-95d5a161-$i/step22200-hf" \
      --tasks "olmo3:dev:1b:main:hf" \
      --priority high \
      --cluster ai2/jupiter-cirrascale-2 \
      --num-gpus 1 \
      --model-backend hf \
      --model-args dtype=bfloat16 \
      --dashboard regmixer \
      --workspace ai2/dolma2

      echo "Completed checkpoint $i"
    } &
done

# Wait for all remaining jobs to complete
wait
echo "All evaluation tasks completed!"