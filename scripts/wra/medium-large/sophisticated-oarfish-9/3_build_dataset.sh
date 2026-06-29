#!/bin/bash
# Stage 3: Build diffusion-ready datasets from completed PD expert runs.
#
# Converts primal history from all 20 PD runs (4 densities x 5 r_min) into
# the raw/ format consumed by WRADataset.process().
#
# For each density, r_min=0.6 is built first with H_instantaneous enabled
# (the reference).  The remaining r_min values symlink their h_instantaneous/
# directories to that reference via link_h_instantaneous_ref.sh.
#
# After Stage 2 completes, fill in the INPUT_DIR paths below from the PD
# output directories (the ones containing collected_samples.npz).
#
# Usage:
#   bash scripts/wra/medium-large/sophisticated-oarfish-9/3_build_dataset.sh

set -euo pipefail
export HYDRA_FULL_ERROR=1

LINK_HELPER=scripts/wra/link_h_instantaneous_ref.sh

COMMON_ARGS=(
    collection.sample_source=primal_history
    collection.primal_history.window_size=1000
    collection.primal_history.refine_feasible_subset=true
    collection.target_samples_per_network=200
)

# ── PD output paths lookup table ──
# Each entry: "density|r_min|input_dir"
# Fill in the input_dir after Stage 2 completes.
# r_min=0.6 MUST appear first within each density block (H_instantaneous reference).
read -r -d '' PD_RUNS <<'TABLE' || true
high|0.6|outputs/wra_medium_outdoor_high_density/wrpd_v1_wrach_v1_s42_D128_N200_R4100_v3_h8aadb4e5a934_r0.6_a0.5_hf8d9aa479b21/2026-05-03/10-16-09
TABLE

# Helper: find the most recently created raw/ dir under a scenario root.
find_latest_raw() {
    local scenario_root="$1"
    find "$scenario_root" -maxdepth 2 -name raw -type d -printf '%T@ %p\n' \
        | sort -n | tail -1 | cut -d' ' -f2-
}

prev_density=""
total=$(echo "$PD_RUNS" | grep -c '|')
counter=0

while IFS='|' read -r density r_min input_dir; do
    [[ -z "$density" ]] && continue

    scenario_root="data/wra/medium_outdoor_${density}_density"
    config="pd_collection/wra_medium_outdoor_${density}_density"
    counter=$(( counter + 1 ))

    # Validate input dir exists
    if [[ ! -d "$input_dir" ]]; then
        echo "ERROR: input_dir not found: $input_dir" >&2
        exit 1
    fi

    is_reference=false
    if [[ "$density" != "$prev_density" ]]; then
        # First entry for this density must be r_min=0.6 (the reference)
        if [[ "$r_min" != "0.6" ]]; then
            echo "ERROR: first r_min for ${density} density must be 0.6, got ${r_min}" >&2
            exit 1
        fi
        is_reference=true
        prev_density="$density"
    fi

    echo "====== [${counter}/${total}] ${density} density | r_min=${r_min} ======"

    if $is_reference; then
        # Build reference with H_instantaneous enabled
        python -m pdi.cli.wra.build_diffusion_dataset \
            --config-name="$config" \
            input_dir="$input_dir" \
            training.r_min="$r_min" \
            output.h_instantaneous.enabled=true \
            "${COMMON_ARGS[@]}"

        # Create the stable anchor symlink for this density
        ref_raw=$(find_latest_raw "$scenario_root")
        bash "$LINK_HELPER" create-anchor "$scenario_root" "$ref_raw"
    else
        # Build without H_instantaneous, then symlink to reference
        python -m pdi.cli.wra.build_diffusion_dataset \
            --config-name="$config" \
            input_dir="$input_dir" \
            training.r_min="$r_min" \
            "${COMMON_ARGS[@]}"

        bash "$LINK_HELPER" link "$scenario_root" "$(find_latest_raw "$scenario_root")"
    fi

    echo "====== Done: ${density} density | r_min=${r_min} ======"
    echo
done <<< "$PD_RUNS"

echo "All 20 diffusion datasets built."
