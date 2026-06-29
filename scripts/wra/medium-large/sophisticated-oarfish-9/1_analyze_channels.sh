#!/bin/bash
# Stage 1: Generate wireless networks and analyze channel statistics for all
# 4 medium-large outdoor density scenarios.
#
# Runs analyze_channels sequentially on a single GPU, producing per-scenario
# deployment visualizations and full-power rate CDFs used to guide r_min
# selection.
#
# Usage:
#   bash scripts/wra/medium-large/sophisticated-oarfish-9/1_analyze_channels.sh
#
#   # Override the number of networks per scenario (default from config: 32):
#   bash scripts/wra/medium-large/sophisticated-oarfish-9/1_analyze_channels.sh 8

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

GPU_ID="${GPU_ID:-0}"

DENSITIES=(
    high
)

NUM_NETWORKS_OVERRIDE="${1:-}"
OVERRIDE_ARGS=()
if [[ -n "${NUM_NETWORKS_OVERRIDE}" ]]; then
    OVERRIDE_ARGS=("dataset.num_networks=${NUM_NETWORKS_OVERRIDE}")
    echo "Overriding dataset.num_networks=${NUM_NETWORKS_OVERRIDE}"
fi

for density in "${DENSITIES[@]}"; do
    config="channel_analysis/wra_medium-large_outdoor_${density}_density"
    echo "====== Analyzing: medium-large outdoor ${density} density ======"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m pdi.cli.wra.analyze_channels \
        --config-name="${config}" \
        "${OVERRIDE_ARGS[@]}"
    echo "====== Done: ${density} density ======"
    echo
done
