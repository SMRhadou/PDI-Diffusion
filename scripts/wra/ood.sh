for RMIN in 0.65 0.7 0.75 0.8 0.85; do
    if [ "$RMIN" = "0.65" ]; then
            LAM_INIT=10.0
        else
            LAM_INIT=20.0
        fi
    micromamba run -n gdiff python scripts/wra/diffusion/wra_baselines.py \
        --methods none \
        --dataset wra_medium_outdoor_high_density \
        --pdi-net-checkpoints \
            "outputs/wireless_resource_allocation-wra/score_net_train/2026-06-01_23-20-14 - energy_portgnn_normal/checkpoints/outer_00400.pt" \
        --dt-checkpoints \
            "outputs/wireless_resource_allocation-wra/score_net_train/2026-06-01_16-35-56 - dual_update_per_batch_fast_lam10_continued_2200/checkpoints/outer_03350.pt" \
            "outputs/wireless_resource_allocation-wra/score_net_train/2026-05-23_12-52-11 - dual_update_per_batch_fast_lam10/checkpoints/outer_02200.pt" \
        --pdi-net-hidden 256 \
        --pdi-net-layers 8 \
        --pdi-net-K 2 \
        --inverse-beta 1.0 \
        --dual-step-size 0.05 \
        --dual-lambda-init $LAM_INIT \
        --energy-mc-samples 8 \
        --num-channel-realizations 50 \
        --n-samples-per-input 200 \
        --max-batches 64 \
        --sub-batch 20 \
        --chunk-size 50 \
        --eval-timeslots 500 \
        --num-evolution-trials 50 \
        --dual-lambda-decay 0.0 \
        --r-min-override $RMIN \
        --label "pdi_net_dt_ood${RMIN//.}_K200"
done