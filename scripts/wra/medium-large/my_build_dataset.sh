#!/bin/bash
# Build diffusion-ready raw WRA dataset from completed PD runs (medium-large outdoor high density)
# using primal history as sample source, for r_min sweep [0.4, 0.5, 0.6, 0.7, 0.8].
#
# r_min=0.6 is built first as the H_instantaneous reference; the others symlink to it.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

SCENARIO_ROOT=data/wra/medium_outdoor_high_density
LINK_HELPER=scripts/wra/link_h_instantaneous_ref.sh
CONFIG=pd_collection/wra_medium_outdoor_high_density
COMMON_ARGS=(
    collection.sample_source=primal_history
    collection.primal_history.window_size=1000
    collection.primal_history.refine_feasible_subset=true
    collection.target_samples_per_network=200
)


# =============================================================================
# 1) Reference: r_min=0.6 (with H_instantaneous enabled)
# =============================================================================
# INPUT_DIR=outputs/wra_medium_outdoor_high_density/wrpd_v1_wrach_v1_s42_D128_N200_R4100_v3_h8aadb4e5a934_r0.6_a0.5_hf8d9aa479b21/2026-05-03/10-16-09
# INPUT_DIR=outputs/wra_medium_outdoor_high_density/wrpd_v1_wrach_v1_s42_D128_N200_R4100_v3_h8aadb4e5a934_r0.6_a0.5_hf8d9aa479b21/2026-05-03/10-16-09
INPUT_DIR=outputs/wra_medium_outdoor_high_density/wrpd_v1_wrach_v1_s42_D256_N200_R4100_v3_h18402b045e82_r0.6_a0.5_hd4e8d4499eeb/2026-05-13/12-28-24
python -m pdi.cli.wra.build_diffusion_dataset \
    --config-name="$CONFIG" \
    input_dir=$INPUT_DIR \
    training.r_min=0.6 \
    output.h_instantaneous.enabled=true \
    "${COMMON_ARGS[@]}" \
    "$@"
