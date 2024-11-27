#!/usr/bin/env bash

set -exuo pipefail

local_image="olmo-core-nightly"
beaker_image="petew/olmo-core-nightly"
beaker_workspace="ai2/dolma2"
image_name="olmo-core-nightly"

beaker image pull "${beaker_image}" "${local_image}"

timestamp=$(date "+%Y%m%d%H%M%S")

beaker_user=$(beaker account whoami --format=json | jq -r '.[0].name')
beaker image create "${local_image}" --name "${image_name}-tmp" --workspace "${beaker_workspace}"
beaker image rename "${beaker_user}/${image_name}" "${image_name}-${timestamp}" || true
beaker image rename "${beaker_user}/${image_name}-tmp" "${image_name}"
